# 向量化流程（Vectorize Pipeline）

从 PDF 年报到 Milvus 向量数据库的完整数据处理流程。

## 文件说明

| 文件 | 步骤 | 说明 |
|------|------|------|
| `batch_parse_pdfs.py` | ① PDF→MD | 批量调用 MinerU API 解析 PDF 为 Markdown |
| `md_preprocessor.py` | ② MD清洗 | 修复 MinerU 输出的格式问题（标题、换行等） |
| `md_image_table_analyze.py` | ③ 图片分析 | 调用 VL 模型分析 MD 中的图片/表格并补充描述 |
| `vision_service.py` | 共享 | 视觉 LLM 服务封装（Qwen VL API） |
| `vectorize_to_milvus.py` | ④ 向量入库 | BGE-M3 向量化 + Milvus 存储 + 基金索引构建 |
| `download_model.py` | 工具 | 下载 BGE-M3 / BGE-Reranker 模型到本地 |
| `query_fund_report.py` | 工具 | 直连 Milvus 按基金代码查询报告内容 |

## 数据目录（在项目根目录）

- `annual_reports_2025_funds/` — 原始 PDF 年报
- `markdown_mineru/` — MinerU 解析输出的 Markdown
- `embedding_model/` — BGE 模型文件（~4GB）

## 完整流程

```bash
# 0. 下载模型（首次运行）
python vectorize/download_model.py

# 1. 解析 PDF → Markdown（需要 MinerU Docker 运行中）
python vectorize/batch_parse_pdfs.py

# 2. Markdown 格式清洗
python vectorize/md_preprocessor.py

# 3. 图片/表格分析（需要 QWEN_API_KEY）
python vectorize/md_image_table_analyze.py

# 4. 向量化并存入 Milvus（需要 Milvus 运行中）
python vectorize/vectorize_to_milvus.py

# 查询已入库的基金报告
python vectorize/query_fund_report.py 159103
python vectorize/query_fund_report.py --list
```

## 依赖

```bash
pip install -r vectorize/requirements.txt
```

> **注意**: `FlagEmbedding` 依赖 PyTorch + CUDA GPU。如只有 CPU，请先安装 CPU 版 torch：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r vectorize/requirements.txt
> ```
