"""
GPU电脑上的一体化查询服务
提供embedding + Milvus检索 + Reranker重排的完整查询接口
"""
from __future__ import annotations

import asyncio
import os
import re
import time
import pymysql

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pymilvus import MilvusClient
from pymilvus.client.types import LoadState
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn

from hybrid_search import rrf_fusion, deduplicate_results


# 限制同时占用GPU模型（encode/rerank）的并发数，避免显存峰值叠加导致OOM。
# 6G显存下两个fp16模型常驻已占用较多空间，可通过环境变量调优。
GPU_SEMAPHORE = asyncio.Semaphore(int(os.getenv("GPU_CONCURRENCY", "2")))
BATCH_RERANK_MAX = int(os.getenv("BATCH_RERANK_MAX", "128"))
BATCH_RERANK_WAIT_MS = float(os.getenv("BATCH_RERANK_WAIT_MS", "50"))

class BatchEncoder:
    """动态批处理：收集短时间窗口内的查询，合并为 batch 推理。

    多个并发请求在 max_wait_ms 内到达时，自动合并为一个 batch 调用 model.encode()，
    GPU 一次推理处理多条，避免逐个推理造成的 GPU 利用率低下。

    使用方式：
        encoded = await batch_encoder.encode(query)
        dense = encoded["dense"]   # 稠密向量 (list[float])
        sparse = encoded["sparse"] # 神经稀疏向量 (dict)
    """

    def __init__(self, model, max_wait_ms: float = 50, max_batch: int = 16):
        self.model = model
        self.max_wait = max_wait_ms / 1000
        self.max_batch = max_batch
        self._queue: list = []  # [(查询, future), ...]
        self._lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def encode(self, query: str) -> dict:
        """提交一个查询，返回 encode 结果（可能被 delay 最多 max_wait 等待拼 batch）。"""
        future = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._queue.append((query, future))
            if self._task is None:
                self._task = asyncio.create_task(self._delayed_flush())
            if len(self._queue) >= self.max_batch:
                self._flush_event.set()  # 满 batch 立刻触发
        return await future

    async def _delayed_flush(self):
        try:
            await asyncio.wait_for(self._flush_event.wait(), timeout=self.max_wait)
        except asyncio.TimeoutError:
            pass
        await self._do_flush()

    async def _do_flush(self):
        async with self._lock:
            if not self._queue:
                self._task = None
                self._flush_event.clear()
                return
            batch = self._queue[:]
            self._queue.clear()
            self._task = None
            self._flush_event.clear()

        queries = [q for q, _ in batch]
        # fast tokenizer 和模型实例不是线程安全的，同一实例只允许一次推理。
        async with self._inference_lock:
            async with GPU_SEMAPHORE:
                result = await asyncio.to_thread(
                    self.model.encode,
                    queries,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
        # 分发结果到各自的 future
        dense_list = result["dense_vecs"]
        sparse_list = result.get("lexical_weights", [])
        for i, (_, future) in enumerate(batch):
            future.set_result({
                "dense": dense_list[i].tolist() if i < len(dense_list) else None,
                "sparse": sparse_list[i] if i < len(sparse_list) else None,
            })


class BatchReranker:
    """Reranker 动态批处理：合并多个并发请求的 pairs，一次 compute_score 调用处理。

    使用方式：
        scores = await batch_reranker.compute(pairs)
        # pairs: [[查询, 段落], ...] → scores: [float, ...]
    """

    def __init__(self, reranker, max_wait_ms: float = 50, max_batch: int = 128):
        if max_batch <= 0:
            raise ValueError("max_batch must be greater than 0")
        self.reranker = reranker
        self.max_wait = max_wait_ms / 1000
        self.max_batch = max_batch
        self._queue: list = []  # [(配对列表, future), ...]
        self._lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def compute(self, pairs: list) -> list:
        """提交一批 pairs，返回对应的 scores 列表。"""
        future = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._queue.append((pairs, future))
            total_pairs = sum(len(p) for p, _ in self._queue)
            if self._task is None:
                self._task = asyncio.create_task(self._delayed_flush())
            if total_pairs >= self.max_batch:
                self._flush_event.set()
        return await future

    async def _delayed_flush(self):
        try:
            await asyncio.wait_for(self._flush_event.wait(), timeout=self.max_wait)
        except asyncio.TimeoutError:
            pass
        await self._do_flush()

    async def _do_flush(self):
        async with self._lock:
            if not self._queue:
                self._task = None
                self._flush_event.clear()
                return
            batch = self._queue[:]
            self._queue.clear()
            self._task = None
            self._flush_event.clear()

        # 合并所有请求的 pairs 为一个大的 pairs 列表
        all_pairs = []
        split_points = []  # 记录每个请求的 pairs 数量，用于拆分结果
        for pairs, _ in batch:
            split_points.append(len(pairs))
            all_pairs.extend(pairs)

        # 严格限制单次 compute_score 的输入规模，避免并发突发时合并出超大 batch。
        scores = []
        try:
            for start in range(0, len(all_pairs), self.max_batch):
                pairs_chunk = all_pairs[start:start + self.max_batch]
                async with self._inference_lock:
                    async with GPU_SEMAPHORE:
                        chunk_scores = await asyncio.to_thread(
                            self.reranker.compute_score, pairs_chunk, normalize=True
                        )

                if hasattr(chunk_scores, "tolist"):
                    chunk_scores = chunk_scores.tolist()
                if not isinstance(chunk_scores, list):
                    chunk_scores = [chunk_scores]
                if len(chunk_scores) != len(pairs_chunk):
                    raise RuntimeError(
                        "Reranker score count mismatch: "
                        f"expected {len(pairs_chunk)}, got {len(chunk_scores)}"
                    )
                scores.extend(float(score) for score in chunk_scores)
        except Exception as exc:
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return

        # 按原始请求拆分结果
        offset = 0
        for i, (_, future) in enumerate(batch):
            n = split_points[i]
            future.set_result([float(s) for s in scores[offset:offset + n]])
            offset += n


# 全局对象
model = None
reranker = None
milvus_client = None
batch_encoder: BatchEncoder | None = None
batch_reranker: BatchReranker | None = None
TABLE_PARENT_TABLE = "fund_report_table_parents"
TABLE_CHILD_LINK_TABLE = "fund_report_table_children"


def _wait_for_collection_loaded(client: MilvusClient, collection_name: str,
                                 timeout: float = 60.0, poll_interval: float = 0.5) -> None:
    """轮询collection加载状态，直到完全加载或超时。

    load_collection() 只是发起加载请求，不保证返回时数据已全部载入内存；
    Milvus 加载是异步的，在此期间 search 可能命中不完整索引，返回极低分数。
    """
    start = time.monotonic()
    while True:
        state = client.get_load_state(collection_name)
        if state.get("state") == LoadState.Loaded:
            return
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"collection {collection_name} 加载超时（{timeout}s），当前状态: {state}")
        time.sleep(poll_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global model, reranker, milvus_client, batch_encoder, batch_reranker

    # 启动时加载Embedding模型
    model_path = os.getenv("EMBEDDING_MODEL_PATH", "./embedding_model/bge-m3")
    print(f"加载BGE-M3模型: {model_path}")
    model = BGEM3FlagModel(model_path, use_fp16=True)
    print("✓ BGE-M3模型加载完成")

    # 初始化动态批处理编码器
    batch_encoder = BatchEncoder(
        model,
        max_wait_ms=float(os.getenv("BATCH_ENCODE_WAIT_MS", "50")),
        max_batch=int(os.getenv("BATCH_ENCODE_MAX", "16")),
    )
    print(f"✓ BatchEncoder 初始化完成 (max_wait={batch_encoder.max_wait*1000:.0f}ms, max_batch={batch_encoder.max_batch})")

    # 加载Reranker模型
    reranker_path = os.getenv("RERANKER_MODEL_PATH", "./embedding_model/bge-reranker-v2-m3")
    print(f"加载BGE-Reranker-v2-m3模型: {reranker_path}")
    reranker = FlagReranker(reranker_path, use_fp16=True)
    print("✓ Reranker模型加载完成")

    # 初始化动态批处理重排器
    batch_reranker = BatchReranker(
        reranker,
        max_wait_ms=BATCH_RERANK_WAIT_MS,
        max_batch=BATCH_RERANK_MAX,
    )
    print(f"✓ BatchReranker 初始化完成 (max_wait={batch_reranker.max_wait*1000:.0f}ms, max_batch={batch_reranker.max_batch})")

    # 连接Milvus（Docker内使用 milvus-standalone:19530，裸机运行时仍用 localhost）
    milvus_url = os.getenv("MILVUS_URL", "http://localhost:19595")
    print(f"连接Milvus: {milvus_url}")
    milvus_client = MilvusClient(uri=milvus_url)
    print("✓ Milvus连接成功")

    # 显式加载collection并等待完成，避免collection仍在loading（未完全加载进内存）时
    # search请求命中不完整索引返回异常低分（这是导致偶发性"检索失效"的根因）
    for collection_name in (
        "fund_reports_mineru",
        "fund_index",
    ):
        if milvus_client.has_collection(collection_name):
            print(f"加载collection: {collection_name}")
            milvus_client.load_collection(collection_name)
            _wait_for_collection_loaded(milvus_client, collection_name)
            print(f"✓ {collection_name} 加载完成")

    yield

    # 关闭时清理
    print("服务关闭")


app = FastAPI(title="Fund Query Service", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(10, ge=10)
    # 只接受单基金代码字符串或 None（全局检索）
    filter_fund_code: Optional[str] = None
    collection_name: str = "fund_reports_mineru"
    search_type: str = "hybrid"  # "dense"（稠密）、"sparse"（稀疏）、"hybrid"（混合）
    use_reranker: bool = True


class SearchResult(BaseModel):
    id: str
    fund_code: str
    fund_name: str
    content: str
    chunk_index: int
    header_1: str = ""
    header_2: str = ""
    header_3: str = ""
    header_4: str = ""
    header_5: str = ""
    header_6: str = ""
    parent_table_id: Optional[str] = None
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total: int


class RerankScoreRequest(BaseModel):
    """计算查询与任意文本内容之间的 Reranker 分数。"""
    query: str
    contents: List[str]


class RerankScoreResponse(BaseModel):
    query: str
    scores: List[float]


@app.post("/rerank/score", response_model=RerankScoreResponse)
async def score_reranker(request: RerankScoreRequest):
    """计算任意 query-content 对的标准化 BGE Reranker 分数。"""
    if batch_reranker is None:
        raise HTTPException(status_code=503, detail="Reranker 服务未就绪")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")
    if not request.contents:
        raise HTTPException(status_code=400, detail="contents 不能为空")
    if any(not content.strip() for content in request.contents):
        raise HTTPException(status_code=400, detail="contents 中不能包含空文本")

    pairs = [[query, content] for content in request.contents]
    scores = await batch_reranker.compute(pairs)
    return RerankScoreResponse(query=query, scores=scores)


@app.post("/fund_reports/search", response_model=SearchResponse)
async def search_funds(request: SearchRequest):
    """
    搜索基金报告内容（支持混合检索+Reranker重排）

    完整检索流程：
    1. 使用BGE-M3模型生成查询向量（稠密+神经稀疏）
    2. 根据search_type执行不同检索策略：
       - dense: 仅稠密向量检索（语义相似）
       - sparse: 仅神经稀疏向量检索（学习型关键词匹配，类似SPLADE）
       - hybrid: 混合检索+RRF融合（推荐）
    3. [可选] 使用BGE-Reranker-v2-m3对候选结果重排（推荐）
    4. 返回top_k个最相关结果

    技术说明：
    - 稀疏向量使用BGE-M3的lexical_weights（learned sparse）
    - 非传统BM25统计方法，而是神经网络端到端训练
    - Reranker使用交叉注意力机制，能够更精确地评估相关性
    - 重排可提升5-15%的排序准确性

    本机只需发送查询文本，无需任何模型或向量处理
    """
    if model is None or milvus_client is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    try:
        # 构建过滤条件：兼容单值/列表/None
        filter_expr = _build_fund_code_filter(request.filter_fund_code)

        # 第一阶段：初步检索
        # 如果使用reranker，先获取更多候选结果
        initial_top_k = request.top_k * 3 if request.use_reranker else request.top_k

        if request.search_type == "dense":
            results = await _search_dense(request, filter_expr, initial_top_k)
        elif request.search_type == "sparse":
            results = await _search_sparse(request, filter_expr, initial_top_k)
        elif request.search_type == "hybrid":
            results = await _search_hybrid(request, filter_expr, initial_top_k)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的搜索类型: {request.search_type}")

        # 第二阶段：Reranker重排（可选）
        if request.use_reranker and reranker is not None and len(results) > 0:
            results = await _rerank_results(request.query, results, request.top_k)
        else:
            # 不使用reranker时，截取top_k
            results = results[:request.top_k]

        results = await _replace_table_results_with_parents(results)

        return SearchResponse(
            results=results,
            query=request.query,
            total=len(results)
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


async def _replace_table_results_with_parents(
    results: List[SearchResult],
) -> List[SearchResult]:
    """将命中的表格子块内容替换为完整父表，保留子块 ID 和 chunk_index。"""
    if not results:
        return results

    child_ids = [result.id for result in results if result.id]
    if not child_ids:
        return results

    try:
        table_rows = await asyncio.to_thread(_fetch_parent_tables, child_ids)
    except Exception as exc:
        # 表格父子回查是结果增强；MySQL 短暂异常时仍返回原子 chunk。
        print(f"表格父子回查失败，返回原子 chunk: {exc}")
        return results

    parent_by_child = {
        row["child_chunk_id"]: row
        for row in table_rows
        if row.get("child_chunk_id") and row.get("parent_table_id")
    }

    for result in results:
        table_row = parent_by_child.get(result.id)
        if table_row and table_row.get("content") is not None:
            result.content = table_row["content"]
            result.parent_table_id = table_row["parent_table_id"]
    return results


def _fetch_parent_tables(child_ids: List[str]) -> List[dict]:
    """批量通过 MySQL 子块关联回查完整父表。"""
    unique_child_ids = list(dict.fromkeys(child_ids))
    placeholders = ", ".join(["%s"] * len(unique_child_ids))
    query = f"""
        SELECT
            child.child_chunk_id,
            child.parent_table_id,
            parent.content
        FROM {TABLE_CHILD_LINK_TABLE} AS child
        INNER JOIN {TABLE_PARENT_TABLE} AS parent
            ON parent.parent_table_id = child.parent_table_id
        WHERE child.child_chunk_id IN ({placeholders})
    """
    with pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "fund_analyser"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=int(os.getenv("MYSQL_CONNECT_TIMEOUT", "5")),
        read_timeout=int(os.getenv("MYSQL_READ_TIMEOUT", "5")),
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, unique_child_ids)
            return cursor.fetchall()


def _build_fund_code_filter(filter_fund_code: Optional[str]) -> Optional[str]:
    """构建 Milvus 标量过滤表达式

    - None / 空字符串 → 不过滤（全局检索）
    - 字符串 → fund_code == "xxx"

    code 必须严格匹配 6 位数字，否则视为非法输入直接拒绝，
    避免恶意输入拼接出任意 Milvus 过滤表达式（filter expression injection）。
    """
    if not filter_fund_code:
        return None
    code = filter_fund_code.strip()
    if not code:
        return None
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError(f"Invalid fund code: {code!r}")
    return f'fund_code == "{code}"'


async def _search_dense(request: SearchRequest, filter_expr: Optional[str], top_k: int = None) -> List[SearchResult]:
    """稠密向量检索"""
    if top_k is None:
        top_k = request.top_k

    # 生成稠密向量（走动态批处理，多个并发请求自动合并为一次 GPU 推理）
    encoded = await batch_encoder.encode(request.query)
    query_embedding = encoded["dense"]

    # 搜索
    search_results = await asyncio.to_thread(
        milvus_client.search,
        collection_name=request.collection_name,
        data=[query_embedding],
        anns_field="embedding",
        limit=top_k,
        output_fields=["id", "fund_code", "fund_name", "content", "chunk_index",
                      "header_1", "header_2", "header_3", "header_4", "header_5", "header_6"],
        filter=filter_expr
    )

    return _format_results(search_results)


async def _search_sparse(request: SearchRequest, filter_expr: Optional[str], top_k: int = None) -> List[SearchResult]:
    """神经稀疏向量检索 (Learned Sparse Retrieval)"""
    if top_k is None:
        top_k = request.top_k

    # 生成稀疏向量（走动态批处理，多个并发请求自动合并为一次 GPU 推理）
    encoded = await batch_encoder.encode(request.query)
    query_sparse = encoded["sparse"]

    # 搜索
    search_results = await asyncio.to_thread(
        milvus_client.search,
        collection_name=request.collection_name,
        data=[query_sparse],
        anns_field="sparse_embedding",
        limit=top_k,
        output_fields=["id", "fund_code", "fund_name", "content", "chunk_index",
                      "header_1", "header_2", "header_3", "header_4", "header_5", "header_6"],
        filter=filter_expr
    )

    return _format_results(search_results)


async def _search_hybrid(request: SearchRequest, filter_expr: Optional[str], top_k: int = None) -> List[SearchResult]:
    """混合检索：稠密+神经稀疏 RRF融合"""
    if top_k is None:
        top_k = request.top_k

    # 同时生成稠密和神经稀疏向量（走动态批处理，多个并发请求自动合并为一次 GPU 推理）
    encoded = await batch_encoder.encode(request.query)
    query_dense = encoded["dense"]
    query_sparse = encoded["sparse"]

    # 获取2倍结果用于融合
    top_k_expanded = top_k * 2
    output_fields = ["id", "fund_code", "fund_name", "content", "chunk_index",
                      "header_1", "header_2", "header_3", "header_4", "header_5", "header_6"]

    # 稠密+稀疏检索真正并发执行（各自丢线程池，避免阻塞事件循环）
    dense_results, sparse_results = await asyncio.gather(
        asyncio.to_thread(
            milvus_client.search,
            collection_name=request.collection_name,
            data=[query_dense],
            anns_field="embedding",
            limit=top_k_expanded,
            output_fields=output_fields,
            filter=filter_expr
        ),
        asyncio.to_thread(
            milvus_client.search,
            collection_name=request.collection_name,
            data=[query_sparse],
            anns_field="sparse_embedding",
            limit=top_k_expanded,
            output_fields=output_fields,
            filter=filter_expr
        ),
    )

    # 格式化为字典列表
    dense_list = _format_results_to_dicts(dense_results)
    sparse_list = _format_results_to_dicts(sparse_results)

    # RRF融合
    fused_results = rrf_fusion(
        dense_results=dense_list,
        sparse_results=sparse_list,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5
    )

    # 去重
    dedup_results = deduplicate_results(fused_results)

    # 转换为SearchResult对象
    final_results = []
    for item in dedup_results[:top_k]:
        final_results.append(SearchResult(
            id=item.get("id", ""),
            fund_code=item.get("fund_code", ""),
            fund_name=item.get("fund_name", ""),
            content=item.get("content", ""),
            chunk_index=item.get("chunk_index", 0),
            header_1=item.get("header_1", ""),
            header_2=item.get("header_2", ""),
            header_3=item.get("header_3", ""),
            header_4=item.get("header_4", ""),
            header_5=item.get("header_5", ""),
            header_6=item.get("header_6", ""),
            score=item.get("score", 0.0)
        ))

    return final_results


async def _rerank_results(query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
    """
    使用BGE-Reranker-v2-m3对候选结果重排

    Args:
        query: 查询文本
        results: 候选结果列表（通常是初步检索的top 20-40）
        top_k: 最终返回的数量

    Returns:
        重排后的top_k结果
    """
    if not results:
        return results

    # 准备reranker输入：[[query, passage], ...]
    pairs = [[query, result.content] for result in results]

    # 调用reranker计算分数（走动态批处理，多个并发请求的 pairs 自动合并为一次 GPU 推理）
    scores = await batch_reranker.compute(pairs)

    # 将新分数附加到结果上
    for result, score in zip(results, scores):
        result.score = float(score)

    # 按新分数排序
    results.sort(key=lambda x: x.score, reverse=True)

    # 返回top_k
    return results[:top_k]


def _format_results(search_results) -> List[SearchResult]:
    """格式化Milvus搜索结果为SearchResult对象"""
    formatted_results = []
    for hits in search_results:
        for hit in hits:
            entity = hit.get("entity", hit)
            formatted_results.append(SearchResult(
                id=entity.get("id", ""),
                fund_code=entity.get("fund_code", ""),
                fund_name=entity.get("fund_name", ""),
                content=entity.get("content", ""),
                chunk_index=entity.get("chunk_index", 0),
                header_1=entity.get("header_1", ""),
                header_2=entity.get("header_2", ""),
                header_3=entity.get("header_3", ""),
                header_4=entity.get("header_4", ""),
                header_5=entity.get("header_5", ""),
                header_6=entity.get("header_6", ""),
                score=hit.get("distance", 0.0)
            ))
    return formatted_results


def _format_results_to_dicts(search_results) -> List[dict]:
    """格式化Milvus搜索结果为字典列表（用于RRF融合）"""
    formatted_results = []
    for hits in search_results:
        for hit in hits:
            entity = hit.get("entity", hit)
            formatted_results.append({
                "id": entity.get("id", ""),
                "fund_code": entity.get("fund_code", ""),
                "fund_name": entity.get("fund_name", ""),
                "content": entity.get("content", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "header_1": entity.get("header_1", ""),
                "header_2": entity.get("header_2", ""),
                "header_3": entity.get("header_3", ""),
                "header_4": entity.get("header_4", ""),
                "header_5": entity.get("header_5", ""),
                "header_6": entity.get("header_6", ""),
                "score": hit.get("distance", 0.0)
            })
    return formatted_results


class FundIdentifyRequest(BaseModel):
    query: str
    top_k: int = 5
    collection_name: str = "fund_index"
    min_score: Optional[float] = 0.5  # 识别置信度阈值，低于此值的不返回


class FundIdentifyResult(BaseModel):
    fund_code: str
    full_name: str
    short_name: str
    score: float


class FundIdentifyResponse(BaseModel):
    results: List[FundIdentifyResult]
    query: str
    total: int


@app.post("/fund_index/search", response_model=FundIdentifyResponse)
async def identify_funds(request: FundIdentifyRequest):
    """
    两级RAG第一级：从用户问题中语义识别基金代码。

    与 /fund_reports/search 不同，这里检索的是 fund_index collection（每只基金一条记录），
    返回的是基金代码，而非年报内容。

    典型用法：
      query="汇添富金融科技ETF的持仓" → 返回 [{"fund_code":"159103", score:0.92}]
      然后用 fund_code 作为 filter 去调用 /fund_reports/search 做第二级年报内容检索。

    注意：
      - 6位数字代码（如"159103"）会先尝试精确匹配（Milvus query，等值过滤），
        命中则直接返回 score=1.0，不受语义检索的召回率影响
      - 精确匹配未命中时（或非纯数字查询），走稠密向量语义检索，
        处理模糊称呼、别名、不带代码的语义匹配场景
    """
    if model is None or milvus_client is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    try:
        query_stripped = request.query.strip()

        # 纯6位数字代码：先尝试精确匹配，保证必中，不依赖语义相似度
        if re.fullmatch(r"\d{6}", query_stripped):
            exact_hits = await asyncio.to_thread(
                milvus_client.query,
                collection_name=request.collection_name,
                filter=f'fund_code == "{query_stripped}"',
                output_fields=["fund_code", "full_name", "short_name"],
                limit=request.top_k,
            )
            if exact_hits:
                results = [
                    FundIdentifyResult(
                        fund_code=hit.get("fund_code", ""),
                        full_name=hit.get("full_name", ""),
                        short_name=hit.get("short_name", ""),
                        score=1.0,
                    )
                    for hit in exact_hits
                ]
                return FundIdentifyResponse(results=results, query=request.query, total=len(results))

        # 用稠密向量做语义检索（走动态批处理）
        encoded = await batch_encoder.encode(request.query)
        query_embedding = encoded["dense"]

        search_results = await asyncio.to_thread(
            milvus_client.search,
            collection_name=request.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            limit=request.top_k,
            output_fields=["fund_code", "full_name", "short_name"],
        )

        results = []
        for hits in search_results:
            for hit in hits:
                score = float(hit.get("distance", 0.0))
                if request.min_score is not None and score < request.min_score:
                    continue
                entity = hit.get("entity", hit)
                results.append(FundIdentifyResult(
                    fund_code=entity.get("fund_code", ""),
                    full_name=entity.get("full_name", ""),
                    short_name=entity.get("short_name", ""),
                    score=score,
                ))

        return FundIdentifyResponse(results=results, query=request.query, total=len(results))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"基金识别失败: {str(e)}")


@app.get("/fund_index/list")
async def list_fund_index(collection_name: str = "fund_index"):
    """列出 fund_index 中所有基金（代码+名称）"""
    if milvus_client is None:
        raise HTTPException(status_code=503, detail="Milvus未连接")
    try:
        results = await asyncio.to_thread(
            milvus_client.query,
            collection_name=collection_name,
            filter="fund_code != ''",
            output_fields=["fund_code", "full_name", "short_name"],
            limit=10000,
        )
        funds = [
            {"code": r["fund_code"], "full_name": r["full_name"], "short_name": r["short_name"]}
            for r in results
        ]
        funds.sort(key=lambda x: x["code"])
        return {"funds": funds, "total": len(funds)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/stats")
async def collection_stats(collection_name: str = "fund_reports_mineru"):
    """获取集合统计信息"""
    if milvus_client is None:
        raise HTTPException(status_code=503, detail="Milvus未连接")

    try:
        stats = await asyncio.to_thread(milvus_client.get_collection_stats, collection_name)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@app.get("/funds")
async def list_funds(collection_name: str = "fund_reports_mineru"):
    """
    获取所有已索引的基金清单（代码 + 名称）

    返回示例：
    {
        "funds": [
            {"code": "159103", "name": "金融科技ETF汇添富"},
            ...
        ],
        "total": 100
    }
    """
    if milvus_client is None:
        raise HTTPException(status_code=503, detail="Milvus未连接")

    try:
        # 查询所有 fund_code 和 fund_name（去重）
        results = await asyncio.to_thread(
            milvus_client.query,
            collection_name=collection_name,
            filter="fund_code != ''",
            output_fields=["fund_code", "fund_name"],
            limit=10000,  # 足够覆盖所有基金
        )

        # 去重：按 fund_code 聚合（一只基金有多个 chunk，只取一条）
        seen = {}
        for r in results:
            code = r.get("fund_code", "").strip()
            name = r.get("fund_name", "").strip()
            if code and code not in seen:
                seen[code] = name

        funds = [{"code": k, "name": v} for k, v in sorted(seen.items())]
        return {"funds": funds, "total": len(funds)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询基金列表失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "reranker_loaded": reranker is not None,
        "milvus_connected": milvus_client is not None
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Fund Query Service (一体化查询服务)",
        "description": "在GPU服务器上完成embedding+向量检索+Reranker重排，本机只需发送查询文本",
        "models": {
            "embedding": "BGE-M3 (稠密+神经稀疏向量)",
            "reranker": "BGE-Reranker-v2-m3"
        },
        "endpoints": {
            "/fund_reports/search": "POST - 查询基金报告（推荐）",
            "/fund_index/search": "POST - 从名称/别名/模糊描述识别基金代码",
            "/stats": "GET - 获取集合统计信息",
            "/health": "GET - 健康检查",
            "/docs": "GET - API交互文档"
        }
    }


if __name__ == "__main__":
    print("="*50)
    print("正在启动 Fund Query Service...")
    print("="*50)

    try:
        # 监听所有网络接口，允许远程访问
        uvicorn.run(
            app,
            host="0.0.0.0",  # 允许外部访问
            port=8001,
            log_level="info"
        )
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
