"""MCP 工具适配器 - 将 MCP 工具转换为 LangChain 工具。

自动发现 MCP 服务器的工具，并转换为 LangChain 可用的格式。
"""
import json
import logging
from typing import Any, Callable, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from app.services.mcp_client import get_mcp_client

logger = logging.getLogger(__name__)


def _json_type_to_python(json_type: str) -> type:
    """JSON Schema 单个类型字符串 → Python 类型。"""
    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    return type_map.get(json_type, str)


def _resolve_field_type(field_info: dict[str, Any]) -> Any:
    """从 JSON Schema 字段定义解析 Python 类型，支持 oneOf/anyOf 与 type 列表。

    用于把 MCP Tool inputSchema 转成 Pydantic 模型字段类型：
    - 简单：{"type": "string"}                → str
    - 联合：{"type": ["string", "array"]}     → str | list
    - oneOf：{"oneOf": [{"type":"string"}, {"type":"array"}, {"type":"null"}]}
                                              → str | list | None
    - 缺省（含 anyOf/oneOf 但所有子项无 type）→ Any
    """
    candidates: list[str] = []

    for key in ("oneOf", "anyOf"):
        for sub in field_info.get(key, []) or []:
            sub_type = sub.get("type") if isinstance(sub, dict) else None
            if isinstance(sub_type, str):
                candidates.append(sub_type)
            elif isinstance(sub_type, list):
                candidates.extend(t for t in sub_type if isinstance(t, str))

    raw_type = field_info.get("type")
    if isinstance(raw_type, str):
        candidates.append(raw_type)
    elif isinstance(raw_type, list):
        candidates.extend(t for t in raw_type if isinstance(t, str))

    # 去重保序
    seen = set()
    unique = [t for t in candidates if not (t in seen or seen.add(t))]

    if not unique:
        # 完全没有 type 信息（例如纯 oneOf 子项没声明 type）→ 放宽到 Any
        return Any

    has_null = "null" in unique
    non_null = [t for t in unique if t != "null"]

    py_types = [_json_type_to_python(t) for t in non_null]
    # 去重，保持顺序
    seen_py = set()
    py_types = [t for t in py_types if not (t in seen_py or seen_py.add(t))]

    if not py_types:
        return type(None)
    if len(py_types) == 1:
        resolved = py_types[0]
    else:
        resolved = Union[tuple(py_types)]  # type: ignore[valid-type]

    return Union[resolved, None] if has_null else resolved  # type: ignore[valid-type]


def _create_pydantic_model_from_schema(
    tool_name: str,
    schema: dict[str, Any],
) -> type[BaseModel]:
    """根据 JSON Schema 创建 Pydantic 模型。

    Args:
        tool_name: 工具名称
        schema: JSON Schema（inputSchema）

    Returns:
        动态创建的 Pydantic 模型类
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # 构建字段定义
    fields: dict[str, Any] = {}
    for field_name, field_info in properties.items():
        base_type = _resolve_field_type(field_info)
        field_description = field_info.get("description", "")

        if field_name in required:
            fields[field_name] = (
                base_type,
                Field(..., description=field_description),
            )
        else:
            default_value = field_info.get("default", None)
            # 非必填字段类型放宽为可空，便于 LLM 省略
            optional_type = base_type if base_type is Any else Union[base_type, None]  # type: ignore[valid-type]
            if default_value is not None:
                fields[field_name] = (
                    optional_type,
                    Field(default_value, description=field_description),
                )
            else:
                fields[field_name] = (
                    optional_type,
                    Field(None, description=field_description),
                )

    # 动态创建模型
    model_name = f"{tool_name.title().replace('_', '')}Input"
    return create_model(model_name, **fields)


async def create_langchain_tool_from_mcp(
    tool_name: str,
    tool_description: str,
    server_name: str,
    input_schema: dict[str, Any],
) -> StructuredTool:
    """从 MCP 工具定义创建 LangChain 工具。
    
    Args:
        tool_name: 工具名称
        tool_description: 工具描述
        server_name: MCP 服务器名称
        input_schema: 工具的输入 JSON Schema
    
    Returns:
        LangChain StructuredTool
    """
    # 创建参数模型
    args_schema = _create_pydantic_model_from_schema(tool_name, input_schema)
    
    # 创建工具函数
    # `config: RunnableConfig` 由 LangChain 自动注入当前运行的 ambient config
    # （见 langchain_core.tools.base._get_runnable_config_param），从而取到
    # chat.py 中写入 config["configurable"]["user_id"] 的当前用户 id，
    # 用于按用户分桶限流，无需改动 agent/graph 的任何调用链。
    async def tool_func(config: RunnableConfig, **kwargs: Any) -> str:
        """动态生成的 MCP 工具调用函数。"""
        try:
            # 过滤掉值为 None 的参数（MCP SDK 会对 None 进行严格验证）
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            user_id = (config.get("configurable") or {}).get("user_id")

            mcp_client = await get_mcp_client()
            result = await mcp_client.call_tool(
                tool_name=tool_name,
                arguments=filtered_kwargs,
                server_name=server_name,
                user_id=user_id,
            )
            
            # 格式化返回结果
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"MCP tool {tool_name} failed: {e}")
            return f"工具调用失败: {str(e)}"
    
    # 创建 LangChain 工具（metadata 携带 server_name 供 agent 按 server 过滤）
    return StructuredTool(
        name=tool_name,
        description=tool_description,
        func=tool_func,
        coroutine=tool_func,
        args_schema=args_schema,
        metadata={"server": server_name},
    )


async def load_mcp_tools() -> list[StructuredTool]:
    """加载所有 MCP 工具为 LangChain 工具列表。
    
    Returns:
        LangChain 工具列表
    """
    mcp_client = await get_mcp_client()
    
    # 获取所有工具
    mcp_tools = await mcp_client.list_all_tools()
    
    if not mcp_tools:
        logger.warning("No MCP tools found")
        return []
    
    # 转换为 LangChain 工具
    langchain_tools = []
    for tool_info in mcp_tools:
        try:
            tool = await create_langchain_tool_from_mcp(
                tool_name=tool_info["name"],
                tool_description=tool_info["description"],
                server_name=tool_info["server"],
                input_schema=tool_info["input_schema"],
            )
            langchain_tools.append(tool)
            logger.info(f"✓ Loaded MCP tool: {tool_info['name']} from {tool_info['server']}")
        except Exception as e:
            logger.error(f"✗ Failed to load tool {tool_info['name']}: {e}")
    
    return langchain_tools
