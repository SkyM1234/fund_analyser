"""
混合检索模块：稠密向量 + 神经稀疏向量 + RRF融合

提供：
1. 神经稀疏检索（使用BGE-M3的学习型稀疏向量，类似SPLADE）
2. 稠密向量检索（原有功能）
3. RRF (Reciprocal Rank Fusion) 融合算法
4. 结果去重

技术说明：
- 稀疏向量使用BGE-M3的lexical_weights（learned sparse retrieval）
- 非传统BM25统计方法，而是端到端训练的神经网络
- 能够学习语义扩展，效果优于传统BM25
"""
from typing import List, Dict
import numpy as np
from collections import defaultdict


def rrf_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5
) -> List[Dict]:
    """
    使用RRF (Reciprocal Rank Fusion) 融合稠密和稀疏检索结果
    
    RRF公式：score = 1 / (k + rank)
    其中k是常数（通常60），rank是结果的排名（从1开始）
    
    Args:
        dense_results: 稠密向量检索结果，格式 [{"id": str, "score": float, ...}, ...]
        sparse_results: 稀疏向量检索结果，格式相同
        k: RRF常数，默认60
        dense_weight: 稠密结果权重
        sparse_weight: 稀疏结果权重
        
    Returns:
        融合后的结果列表，按RRF分数降序排列
    """
    # 收集所有文档的RRF分数
    rrf_scores = defaultdict(float)
    doc_data = {}  # 存储文档完整数据
    
    # 处理稠密向量结果
    for rank, result in enumerate(dense_results, start=1):
        doc_id = result.get("id") or f"{result.get('fund_code')}_{result.get('chunk_index')}"
        rrf_scores[doc_id] += dense_weight * (1.0 / (k + rank))
        if doc_id not in doc_data:
            doc_data[doc_id] = result.copy()
    
    # 处理稀疏向量结果
    for rank, result in enumerate(sparse_results, start=1):
        doc_id = result.get("id") or f"{result.get('fund_code')}_{result.get('chunk_index')}"
        rrf_scores[doc_id] += sparse_weight * (1.0 / (k + rank))
        if doc_id not in doc_data:
            doc_data[doc_id] = result.copy()
    
    # 按RRF分数排序
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 构建最终结果
    final_results = []
    for doc_id, rrf_score in sorted_docs:
        result = doc_data[doc_id].copy()
        result["score"] = rrf_score  # 使用RRF分数替换原始分数
        result["fusion_method"] = "rrf"
        final_results.append(result)
    
    return final_results


def deduplicate_results(results: List[Dict], key_fields: List[str] = None) -> List[Dict]:
    """
    对检索结果去重
    
    Args:
        results: 检索结果列表
        key_fields: 用于判断重复的字段，默认使用 ["fund_code", "chunk_index"]
        
    Returns:
        去重后的结果列表
    """
    if key_fields is None:
        key_fields = ["fund_code", "chunk_index"]
    
    seen = set()
    deduplicated = []
    
    for result in results:
        # 构建唯一键
        key_values = tuple(result.get(field, "") for field in key_fields)
        
        if key_values not in seen:
            seen.add(key_values)
            deduplicated.append(result)
    
    return deduplicated


def normalize_scores(results: List[Dict], method: str = "minmax") -> List[Dict]:
    """
    归一化检索分数到[0, 1]区间
    
    Args:
        results: 检索结果列表
        method: 归一化方法，支持 "minmax" 或 "zscore"
        
    Returns:
        归一化后的结果列表
    """
    if not results:
        return results
    
    scores = np.array([r.get("score", 0.0) for r in results])
    
    if method == "minmax":
        # Min-Max归一化
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores)
    
    elif method == "zscore":
        # Z-score归一化
        mean_score = scores.mean()
        std_score = scores.std()
        if std_score > 0:
            normalized_scores = (scores - mean_score) / std_score
            # 映射到[0, 1]
            normalized_scores = 1 / (1 + np.exp(-normalized_scores))
        else:
            normalized_scores = np.ones_like(scores)
    
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    # 更新结果中的分数
    normalized_results = []
    for result, norm_score in zip(results, normalized_scores):
        result_copy = result.copy()
        result_copy["score"] = float(norm_score)
        normalized_results.append(result_copy)
    
    return normalized_results


def weighted_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3
) -> List[Dict]:
    """
    使用加权平均融合稠密和稀疏检索结果（分数归一化后加权）
    
    Args:
        dense_results: 稠密向量检索结果
        sparse_results: 稀疏向量检索结果
        dense_weight: 稠密结果权重
        sparse_weight: 稀疏结果权重
        
    Returns:
        融合后的结果列表
    """
    # 先归一化分数
    dense_normalized = normalize_scores(dense_results, method="minmax")
    sparse_normalized = normalize_scores(sparse_results, method="minmax")
    
    # 收集所有文档的加权分数
    weighted_scores = defaultdict(float)
    doc_data = {}
    
    # 处理稠密结果
    for result in dense_normalized:
        doc_id = result.get("id") or f"{result.get('fund_code')}_{result.get('chunk_index')}"
        weighted_scores[doc_id] += dense_weight * result.get("score", 0.0)
        if doc_id not in doc_data:
            doc_data[doc_id] = result.copy()
    
    # 处理稀疏结果
    for result in sparse_normalized:
        doc_id = result.get("id") or f"{result.get('fund_code')}_{result.get('chunk_index')}"
        weighted_scores[doc_id] += sparse_weight * result.get("score", 0.0)
        if doc_id not in doc_data:
            doc_data[doc_id] = result.copy()
    
    # 排序
    sorted_docs = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 构建结果
    final_results = []
    for doc_id, weighted_score in sorted_docs:
        result = doc_data[doc_id].copy()
        result["score"] = weighted_score
        result["fusion_method"] = "weighted"
        final_results.append(result)
    
    return final_results
