# Graph可视化工具

## 概述

Multi-Agent架构的graph可视化工具，可以导出为Mermaid图表或PNG图片，便于理解和文档化系统架构。

## 功能

### 1. 打印Graph结构

打印所有节点和边的列表，用于快速查看架构。

```python
from app.agent.multi_agent_controller import print_graph_structure

print_graph_structure()
```

**输出示例**：
```
============================================================
Multi-Agent Graph Structure
============================================================

📦 Nodes (11):
  - __start__
  - route
  - supervisor
  - dispatcher
  - rag_agent
  - market_agent
  - reflection
  - synthesizer
  - compliance
  - compliance_failure_handler
  - __end__

🔗 Edges:
  __start__ → route
  route → supervisor
  supervisor → dispatcher
  dispatcher → rag_agent
  dispatcher → market_agent
  rag_agent → reflection
  market_agent → reflection
  reflection → dispatcher
  reflection → synthesizer
  synthesizer → compliance
  compliance → __end__
  ...
============================================================
```

### 2. 导出Mermaid图表

导出为Mermaid语法的文本文件，可在Markdown中渲染。

```python
from app.agent.multi_agent_controller import export_graph_to_mermaid

# 导出到文件
mermaid_str = export_graph_to_mermaid(output_file="docs/graph_mermaid.md")

# 或获取字符串
mermaid_str = export_graph_to_mermaid()
print(mermaid_str)
```

### 3. 导出PNG图片

导出为PNG图片（需要安装额外依赖）。

```python
from app.agent.multi_agent_controller import export_graph_to_png

# 导出到文件
success = export_graph_to_png(output_file="docs/graph_architecture.png")
```

**依赖安装**：
```bash
# 方案1：使用LangGraph内置方法（推荐）
pip install langgraph

# 方案2：使用Graphviz（可选，更精细控制）
# Windows: 下载并安装 https://graphviz.org/download/
# 然后安装Python包
pip install pygraphviz
```

## 命令行工具

直接运行文件即可生成所有可视化：

```bash
cd backend
python -m app.agent.multi_agent_controller
```

**输出**：
```
🎨 Multi-Agent Graph Visualization Tool

📊 Creating graph...

============================================================
Multi-Agent Graph Structure
============================================================
[节点和边的列表]

📄 Exporting Mermaid diagram to backend/docs/graph_mermaid.md...
   ✓ Mermaid diagram exported (2345 chars)

🖼️  Exporting PNG diagram to backend/docs/graph_architecture.png...
   ✓ PNG diagram exported

✅ Done!
```

## 输出文件

执行后会生成：

1. **backend/docs/graph_mermaid.md** - Mermaid图表文本
2. **backend/docs/graph_architecture.png** - PNG架构图（如果支持）

## Mermaid图表示例

生成的Mermaid图表可以在GitHub、GitLab等平台直接渲染：

\`\`\`mermaid
graph TD
    __start__ --> route
    route --> supervisor
    supervisor --> dispatcher
    dispatcher --> rag_agent
    dispatcher --> market_agent
    rag_agent --> reflection
    market_agent --> reflection
    reflection --> dispatcher
    reflection --> synthesizer
    synthesizer --> compliance
    compliance --> __end__
\`\`\`

## 高级用法

### 在代码中集成

```python
from app.agent.multi_agent_controller import (
    build_multi_agent_graph,
    export_graph_to_mermaid,
    export_graph_to_png,
    print_graph_structure
)

# 1. 创建graph
from app.services.checkpoint import get_postgres_checkpointer

checkpointer = await get_postgres_checkpointer()
graph = build_multi_agent_graph(checkpointer)

# 2. 打印结构
print_graph_structure(checkpointer)

# 3. 导出可视化
export_graph_to_mermaid(checkpointer, "custom_path.md")
export_graph_to_png(checkpointer, "custom_path.png")
```

## 故障排查

### Q: PNG导出失败？

**A**: 确保安装了依赖：
```bash
pip install langgraph --upgrade
```

如果仍然失败，检查LangGraph版本：
```bash
pip show langgraph
# 需要 >= 0.2.0
```

### Q: Mermaid图在GitHub不显示？

**A**: 确保文件是`.md`格式，并使用正确的代码块语法：
\`\`\`mermaid
graph TD
    ...
\`\`\`

### Q: 图太复杂看不清？

**A**: 可以修改节点命名使其更简洁，或分模块导出子图。

## 后续改进

- [ ] 支持导出子图（只显示特定路径）
- [ ] 添加节点颜色标注（按类型区分）
- [ ] 支持交互式图表（HTML）
- [ ] 添加性能统计标注（调用次数、耗时）
