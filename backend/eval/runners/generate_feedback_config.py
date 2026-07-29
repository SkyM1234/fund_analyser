"""
自定义生成反馈配置
"""
from __future__ import annotations

import asyncio

from langsmith import Client

from eval.config import get_eval_settings

async def generate_feedback_config():
    s = get_eval_settings()
    s.export_langsmith_env()
    client = Client()
    for config in client.list_feedback_configs():
        print(f"配置详情: {config}") # 打印反馈配置（包含UI不可见的隐式配置）
    
    client.delete_feedback_config(feedback_key="...") # 删除反馈配置（包括UI不可见的隐式配置）
    
    client.create_feedback_config(
    feedback_key="...",
    feedback_config={
        "type": "continuous",
        "min": 0,  # 替换为你的实际分数最小值
        "max": 1   # 替换为你的实际分数最大值
    },
    is_lower_score_better=True  # 正确字段名：实现低分标绿、高分标红
    ) # 创建反馈配置



def main():
    asyncio.run(
        generate_feedback_config()
    )


if __name__ == "__main__":
    main()
