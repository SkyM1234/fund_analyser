"""
使用BGE-M3向量化analyzed markdown文件并存储到Milvus
支持稠密向量 + 神经稀疏向量（Learned Sparse Retrieval）的混合检索
"""
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient, DataType
from langchain_text_splitters import MarkdownHeaderTextSplitter
import hashlib
import warnings

# 抑制transformers的性能警告
warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

# 常量定义
MAX_HEADER_LEVELS = 6  # 最大标题层级数
MIN_SECTION_LENGTH = 500  # 短章节合并的最小长度阈值
TOC_DOT_THRESHOLD = 3  # 目录判定的点号数量阈值
CONTENT_MAX_LENGTH = 3000  # content 字段的最大长度（匹配 Milvus schema），注意Milvus中长度按UTF-8编码计算
OVERLAP_MIN_LENGTH = 20  # overlap 文本的最小有效长度

# 预编译的正则表达式
RE_PAGE_MARKER = re.compile(r'<!--\s*第\s*\d+\s*页\s*-->\s*')
RE_TABLE_START = re.compile(r'<!--\s*TABLE_START[^>]*?-->')
RE_TABLE_END = re.compile(r'<!--\s*TABLE_END[^>]*?-->')
RE_IMAGE_START = re.compile(r'<!--\s*IMAGE_START[^>]*?-->')
RE_IMAGE_END = re.compile(r'<!--\s*IMAGE_END[^>]*?-->')
RE_IMAGE_MARKDOWN = re.compile(r'!\[.*?\]\(.*?\)')
RE_MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
RE_TABLE_BLOCK = re.compile(r'<!--\s*TABLE_START[^>]*?-->.*?<!--\s*TABLE_END[^>]*?-->', re.DOTALL)
RE_IMAGE_BLOCK = re.compile(r'<!--\s*IMAGE_START[^>]*?-->.*?<!--\s*IMAGE_END[^>]*?-->', re.DOTALL)
RE_HEADER_PATTERN = re.compile(r'\n(#{1,6})\s+[^\n]+')
RE_SENTENCE_END = re.compile(r'[。！？；]|(\n\n)')
RE_PUNCTUATION_SPLIT = re.compile(r'([。！？；\n])')


class FundVectorizer:
    def __init__(self,
                 milvus_host: str = "localhost",
                 milvus_port: int = 19595,
                 collection_name: str = "fund_reports_mineru",
                 model_path: str = "./embedding_model/bge-m3"):
        """
        初始化向量化器

        Args:
            milvus_host: Milvus服务器地址
            milvus_port: Milvus端口
            collection_name: 集合名称
            model_path: BGE-M3模型路径（本地路径）
        """
        self.collection_name = collection_name

        # 初始化 MarkdownHeaderTextSplitter
        headers_to_split_on = [(f"{'#' * i}", f"Header {i}") for i in range(1, MAX_HEADER_LEVELS + 1)]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # 检查模型路径
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            print(f"模型路径不存在: {model_path}")
            print("正在从HuggingFace下载模型到本地...")
            print("如果下载失败，请手动下载模型到该路径")
            # 创建目录
            model_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # 加载BGE-M3模型
        print(f"加载BGE-M3模型: {model_path}")
        self.model = BGEM3FlagModel(model_path, use_fp16=True)

        # 连接Milvus
        print(f"连接Milvus: {milvus_host}:{milvus_port}")
        self.client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")

        # 创建或获取集合
        self._setup_collection()

    def _extract_header_path(self, metadata: Dict) -> List[str]:
        """
        从 metadata 中提取标题路径

        Args:
            metadata: 包含 Header 1~6 的字典

        Returns:
            标题路径列表，如 ['§1 重要提示及目录', '1.2 目录']
        """
        header_path_parts = []
        for i in range(1, MAX_HEADER_LEVELS + 1):
            header = metadata.get(f'Header {i}', '')
            if header:
                header_path_parts.append(header)
        return header_path_parts

    def _build_header_prefix(self, header_path_parts: List[str]) -> str:
        """
        构建标题路径前缀字符串

        Args:
            header_path_parts: 标题路径列表

        Returns:
            标题路径前缀，如 '[§1 重要提示及目录 > 1.2 目录]\n\n'
        """
        if header_path_parts:
            header_path = ' > '.join(header_path_parts)
            return f"[{header_path}]\n\n"
        return ""

    def _create_chunk_dict(self, content: str, metadata: Dict) -> Dict:
        """
        创建标准的 chunk 字典

        Args:
            content: chunk 内容
            metadata: 标题元数据

        Returns:
            包含 content 和 header_1~6 的字典
        """
        chunk_dict = {'content': content}
        for i in range(1, MAX_HEADER_LEVELS + 1):
            chunk_dict[f'header_{i}'] = metadata.get(f'Header {i}', '')
        return chunk_dict

    def _setup_collection(self):
        """设置Milvus集合（仅在不存在时创建，不删除已有数据）"""
        if self.client.has_collection(self.collection_name):
            print(f"集合 {self.collection_name} 已存在，跳过创建")
            return

        # 定义集合schema
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False
        )

        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="fund_code", datatype=DataType.VARCHAR, max_length=10)
        schema.add_field(field_name="fund_name", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=9000)  # chunk_size(2000) *3(中文--UTF-8字节) * 1.5(超大表格)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(field_name="header_1", datatype=DataType.VARCHAR, max_length=500)  # 一级标题
        schema.add_field(field_name="header_2", datatype=DataType.VARCHAR, max_length=500)  # 二级标题
        schema.add_field(field_name="header_3", datatype=DataType.VARCHAR, max_length=500)  # 三级标题
        schema.add_field(field_name="header_4", datatype=DataType.VARCHAR, max_length=500)  # 四级标题
        schema.add_field(field_name="header_5", datatype=DataType.VARCHAR, max_length=500)  # 五级标题
        schema.add_field(field_name="header_6", datatype=DataType.VARCHAR, max_length=500)  # 六级标题
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)  # 稠密向量
        schema.add_field(field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 神经稀疏向量(Learned Sparse)

        # 创建索引参数
        index_params = self.client.prepare_index_params()
        # 稠密向量索引
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )
        # 稀疏向量索引 (Learned Sparse Retrieval)
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",  # Inner Product for sparse vectors
            params={"drop_ratio_build": 0.2}
        )

        # 创建集合
        print(f"创建集合 {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        print("集合创建完成")

    def chunk_text_with_headers(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
        """
        使用多阶段分割策略处理文本：
        第一阶段：清理页码标记
        第二阶段：使用 MarkdownHeaderTextSplitter 按标题结构分割
        第三阶段：过滤无价值章节（目录等）
        第四阶段：合并短章节，减少碎片化
        第五阶段：对每个标题块应用智能细分（保护表格和图片完整性）
        第六阶段：为每个 chunk 添加标题路径前缀

        Args:
            text: 输入文本
            chunk_size: 目标块大小（字符数）
            overlap: 重叠大小

        Returns:
            包含内容和标题元数据的字典列表
        """
        # 第一阶段：清理页码标记
        text = self._remove_page_markers(text)

        # 第二阶段：按标题结构分割
        header_chunks = self.markdown_splitter.split_text(text)

        # 第三阶段：过滤无价值章节并标准化数据结构
        filtered_chunks = []
        for header_chunk in header_chunks:
            metadata = header_chunk.metadata
            content = header_chunk.page_content

            # 使用辅助方法构建标题路径
            header_path_parts = self._extract_header_path(metadata)

            # 判断是否为无价值章节，跳过
            if self._should_skip_section(header_path_parts, content):
                continue

            filtered_chunks.append({
                'metadata': metadata,
                'content': content,
                'header_path_parts': header_path_parts
            })

        # 第四阶段：合并短章节
        merged_chunks = self._merge_short_sections(filtered_chunks, min_length=MIN_SECTION_LENGTH)

        result_chunks = []

        # 第五阶段和第六阶段：对每个标题块进行细分并添加标题路径前缀
        for chunk_data in merged_chunks:
            metadata = chunk_data['metadata']
            content = chunk_data['content']
            header_path_parts = chunk_data['header_path_parts']

            # 使用辅助方法构建标题路径前缀
            header_prefix = self._build_header_prefix(header_path_parts)

            # 对内容进行细分
            sub_chunks = self._chunk_content(content, chunk_size, overlap)

            # 将标题路径添加到每个子块的 content 前面
            for sub_chunk in sub_chunks:
                content_with_header = header_prefix + sub_chunk
                # 使用辅助方法创建 chunk 字典
                result_chunks.append(self._create_chunk_dict(content_with_header, metadata))

        return result_chunks

    def _merge_short_sections(self, chunks: List[Dict], min_length: int = 500) -> List[Dict]:
        """
        合并短章节，减少碎片化

        合并策略：
        1. 识别同一父标题下的兄弟章节
        2. 如果相邻的兄弟章节都很短（< min_length），合并它们
        3. 合并时保留第一个章节的标题作为主标题，其他章节的标题作为子标题注入内容

        Args:
            chunks: 标题块列表
            min_length: 最小长度阈值，短于此长度的相邻章节会被合并

        Returns:
            合并后的标题块列表
        """
        if not chunks:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]
            current_length = len(current_chunk['content'])

            # 如果当前章节长度足够，直接保留
            if current_length >= min_length:
                merged.append(current_chunk)
                i += 1
                continue

            # 当前章节很短，尝试与后续的兄弟章节合并
            merge_group = [current_chunk]
            merge_total_length = current_length
            j = i + 1

            # 查找可合并的相邻短章节
            while j < len(chunks) and merge_total_length < min_length:
                next_chunk = chunks[j]

                # 判断是否为兄弟章节（共享相同的父标题）
                if self._are_sibling_sections(current_chunk, next_chunk):
                    # 如果下一个章节也很短，加入合并组
                    if len(next_chunk['content']) < min_length:
                        merge_group.append(next_chunk)
                        merge_total_length += len(next_chunk['content'])
                        j += 1
                    else:
                        # 下一个章节不短，停止合并
                        break
                else:
                    # 不是兄弟章节，停止合并
                    break

            # 如果只有一个章节，即使短也保留
            if len(merge_group) == 1:
                merged.append(current_chunk)
                i += 1
            else:
                # 合并多个短章节
                merged_chunk = self._merge_chunks(merge_group)
                merged.append(merged_chunk)
                i = j

        return merged

    def _are_sibling_sections(self, chunk1: Dict, chunk2: Dict) -> bool:
        """
        判断两个章节是否为兄弟章节（共享相同的父标题）

        Args:
            chunk1: 第一个章节
            chunk2: 第二个章节

        Returns:
            True 表示是兄弟章节，False 表示不是
        """
        path1 = chunk1['header_path_parts']
        path2 = chunk2['header_path_parts']

        # 如果路径长度相同且除了最后一级标题外都相同，则为兄弟章节
        if len(path1) != len(path2) or len(path1) == 0:
            return False

        # 检查父路径是否相同（除最后一级）
        return path1[:-1] == path2[:-1]

    def _merge_chunks(self, chunks: List[Dict]) -> Dict:
        """
        合并多个章节为一个

        Args:
            chunks: 要合并的章节列表

        Returns:
            合并后的章节
        """
        if not chunks:
            return None

        if len(chunks) == 1:
            return chunks[0]

        # 使用父标题作为合并后的主标题（去掉最后一级子标题）
        first_chunk = chunks[0]
        parent_header_path = first_chunk['header_path_parts'][:-1] if len(first_chunk['header_path_parts']) > 1 else first_chunk['header_path_parts']

        # 构建父标题的 metadata
        merged_metadata = {}
        for i, header in enumerate(parent_header_path, 1):
            merged_metadata[f'Header {i}'] = header

        # 合并内容：每个章节的内容前添加其子标题
        merged_content_parts = []
        for chunk in chunks:
            # 如果有子标题（最后一级标题），添加为小节标题
            if chunk['header_path_parts']:
                section_title = chunk['header_path_parts'][-1]
                merged_content_parts.append(f"### {section_title}\n\n{chunk['content']}")
            else:
                merged_content_parts.append(chunk['content'])

        merged_content = '\n\n'.join(merged_content_parts)

        return {
            'metadata': merged_metadata,
            'content': merged_content,
            'header_path_parts': parent_header_path
        }

    def _should_skip_section(self, header_path: List[str], content: str) -> bool:
        """
        判断章节是否应该跳过（无检索价值）

        跳过规则：
        1. 目录章节：标题包含"目录"，内容主要是点号+数字索引

        Args:
            header_path: 标题路径列表，如 ['§1 重要提示及目录', '1.2 目录']
            content: 章节内容

        Returns:
            True 表示应该跳过，False 表示保留
        """
        # 合并标题路径为字符串
        header_text = ' '.join(header_path)

        # 规则1：目录章节
        if '目录' in header_text:
            # 检查内容是否主要是索引格式（大量点号）
            dot_count = content.count('.')
            if dot_count > TOC_DOT_THRESHOLD and content[-1].isdigit():
                return True

        return False

    def _remove_page_markers(self, text: str) -> str:
        """
        移除MD中的独立页码标记，避免干扰 MarkdownHeaderTextSplitter 切分

        移除内容：
        1. 独立页码注释：<!-- 第 38 页 -->

        注意：TABLE_START/IMAGE_START 注释会在后续 _clean_content_for_embedding 中统一清理

        Args:
            text: 输入文本

        Returns:
            清理后的文本
        """
        # 使用预编译的正则表达式移除独立页码注释
        return RE_PAGE_MARKER.sub('', text)

    def _chunk_content(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        对内容进行智能细分，最高优先级保护表格和图片完整性

        算法核心：基于语义单元的贪心打包
        1. 解析文本为有序的语义单元列表（段落/表格/图片）
        2. 贪心打包，含锚点(表格/图片)的 chunk 允许扩展到 hard_max
           这样前置元数据和后置注释能自然跟随锚点
        3. 超大表格切分后：第一块继承前文累积，最后一块作为新 current 让后文累积

        Args:
            text: 输入文本
            chunk_size: 目标块大小（字符数）
            overlap: 重叠大小

        Returns:
            文本块列表
        """
        # 第一步：提取所有表格和图片的位置信息
        structures = self._extract_structures(text)

        if not structures:
            # 没有表格和图片，直接按段落处理
            return self._chunk_plain_text(text, chunk_size, overlap)

        # 第二步：将文本解析为语义单元
        segments = self._parse_segments(text, structures)

        # 第三步：贪心打包语义单元
        return self._pack_segments(segments, text, structures, chunk_size, overlap)

    def _parse_segments(self, text: str, structures: List[Dict]) -> List[Dict]:
        """
        将文本解析为有序的语义单元列表

        Args:
            text: 原文
            structures: 已提取的表格/图片结构列表

        Returns:
            语义单元列表，每项包含：
              - type: 'paragraph' | 'table' | 'image'
              - text: 单元文本
              - start: 在原文中的起始位置
              - end: 在原文中的结束位置
              - is_anchor: 是否为锚点（表格/图片）
        """
        segments = []
        pos = 0

        for struct in structures:
            # 处理结构之前的普通文本
            if struct['start'] > pos:
                plain_text = text[pos:struct['start']]
                # 按双换行切分为段落，保留段落在原文中的位置
                offset = pos
                for raw_para in plain_text.split('\n\n'):
                    para_stripped = raw_para.strip()
                    if para_stripped:
                        # 计算这个段落在原文中的位置
                        para_start = text.find(para_stripped, offset)
                        if para_start == -1:
                            para_start = offset
                        para_end = para_start + len(para_stripped)
                        segments.append({
                            'type': 'paragraph',
                            'text': para_stripped,
                            'start': para_start,
                            'end': para_end,
                            'is_anchor': False
                        })
                        offset = para_end
                    else:
                        offset += len(raw_para) + 2  # +2 for '\n\n'

            # 添加结构单元
            segments.append({
                'type': struct['type'].lower(),  # 'table' or 'image'
                'text': struct['content'],
                'start': struct['start'],
                'end': struct['end'],
                'is_anchor': True
            })
            pos = struct['end']

        # 处理最后剩余的文本
        if pos < len(text):
            plain_text = text[pos:]
            offset = pos
            for raw_para in plain_text.split('\n\n'):
                para_stripped = raw_para.strip()
                if para_stripped:
                    para_start = text.find(para_stripped, offset)
                    if para_start == -1:
                        para_start = offset
                    para_end = para_start + len(para_stripped)
                    segments.append({
                        'type': 'paragraph',
                        'text': para_stripped,
                        'start': para_start,
                        'end': para_end,
                        'is_anchor': False
                    })
                    offset = para_end
                else:
                    offset += len(raw_para) + 2

        return segments

    def _pack_segments(self, segments: List[Dict], text: str, structures: List[Dict],
                       chunk_size: int, overlap: int) -> List[str]:
        """
        贪心打包语义单元为 chunks

        规则：
        1. target_size = chunk_size（普通 chunk 严格遵守）
        2. hard_max = chunk_size * 1.5（含锚点的 chunk 允许扩展）
        3. 普通段落：
           - 能放下（≤ target_size 或含锚点 ≤ hard_max）→ 加入当前 chunk
           - 否则 → flush，开新 chunk（带前向 overlap）
        4. 锚点（表格/图片）：
           - 超大锚点（> hard_max）：调用切分逻辑
           - 普通锚点：能合并就合并（最多扩展到 hard_max），否则开新 chunk

        Args:
            segments: 语义单元列表
            text: 原文（用于获取智能 overlap）
            structures: 结构列表（用于智能 overlap 边界判断）
            chunk_size: 目标块大小
            overlap: 重叠大小

        Returns:
            chunks 列表
        """
        target_size = chunk_size
        hard_max = int(chunk_size * 1.5)
        SEP = '\n\n'
        SEP_LEN = len(SEP)

        chunks: List[str] = []
        current_parts: List[str] = []
        current_length = 0
        current_has_anchor = False
        # 当前 chunk 的起始位置（用于生成下一个 chunk 的 overlap）
        current_start_pos = segments[0]['start'] if segments else 0

        def flush():
            """保存当前累积内容为一个 chunk"""
            nonlocal current_parts, current_length, current_has_anchor
            if current_parts:
                chunk_text = SEP.join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            current_parts = []
            current_length = 0
            current_has_anchor = False

        def append_to_current(seg_text: str, is_anchor: bool = False):
            """将文本追加到当前 chunk"""
            nonlocal current_length, current_has_anchor
            current_parts.append(seg_text)
            # 计算累积长度：所有 parts 用 SEP 连接
            current_length = sum(len(p) for p in current_parts) + SEP_LEN * (len(current_parts) - 1)
            if is_anchor:
                current_has_anchor = True

        def start_new_chunk_with_overlap(seg: Dict, seg_text: str, is_anchor: bool = False):
            """flush 当前 chunk，并用前向 overlap + seg_text 启动新 chunk"""
            nonlocal current_start_pos, current_has_anchor
            flush()
            overlap_text = self._get_smart_overlap_before(text, seg['start'], overlap, structures)
            if overlap_text:
                current_parts.append(overlap_text)
            current_parts.append(seg_text)
            _recompute_length()
            current_start_pos = seg['start']
            if is_anchor:
                current_has_anchor = True

        def _recompute_length():
            nonlocal current_length
            current_length = sum(len(p) for p in current_parts) + SEP_LEN * max(len(current_parts) - 1, 0)

        for seg in segments:
            seg_text = seg['text']
            seg_len = len(seg_text)

            # 计算加入这个 seg 后的总长度（含分隔符）
            projected_length = current_length + (SEP_LEN if current_parts else 0) + seg_len

            if seg['is_anchor']:
                # ===== 锚点处理（表格/图片）=====

                if seg_len > hard_max:
                    # --- 超大锚点 ---
                    if seg['type'] == 'table':
                        sub_chunks = self._split_large_table(seg_text, chunk_size)
                    else:
                        # 超大图片不切分
                        sub_chunks = [seg_text]

                    if not sub_chunks:
                        sub_chunks = [seg_text]

                    # 第一个子块：尝试合并到 current（保留前文累积）
                    first_sub = sub_chunks[0]
                    first_projected = current_length + (SEP_LEN if current_parts else 0) + len(first_sub)
                    if current_parts and first_projected <= hard_max:
                        # 前文累积 + 第一个子块 合并为一个 chunk
                        append_to_current(first_sub, is_anchor=True)
                        flush()
                    else:
                        # 前文太多，无法合并；将前文作为独立 chunk
                        if current_parts:
                            flush()
                        # 第一个子块：带前向 overlap 独立成块
                        overlap_text = self._get_smart_overlap_before(text, seg['start'], overlap, structures)
                        if overlap_text:
                            chunks.append((overlap_text + SEP + first_sub).strip())
                        else:
                            chunks.append(first_sub.strip())

                    # 中间子块：独立成块
                    for sub in sub_chunks[1:-1]:
                        chunks.append(sub.strip())

                    # 最后一个子块：作为新 current chunk，让后续段落累积
                    if len(sub_chunks) > 1:
                        last_sub = sub_chunks[-1]
                        current_parts = [last_sub]
                        current_length = len(last_sub)
                        current_has_anchor = True
                        current_start_pos = seg['start']

                else:
                    # --- 普通大小锚点 ---
                    if not current_parts:
                        # 当前 chunk 为空，直接加入
                        append_to_current(seg_text, is_anchor=True)
                        current_start_pos = seg['start']
                    elif projected_length <= hard_max:
                        # 含锚点允许扩展到 hard_max
                        append_to_current(seg_text, is_anchor=True)
                    else:
                        # 放不下：flush，新 chunk = 前向 overlap + 锚点
                        start_new_chunk_with_overlap(seg, seg_text, is_anchor=True)

            else:
                # ===== 普通段落处理 =====

                # 决定容量上限：含锚点的 chunk 可扩展到 hard_max
                capacity = hard_max if current_has_anchor else target_size

                if not current_parts:
                    # 当前 chunk 为空
                    if seg_len <= target_size:
                        append_to_current(seg_text)
                        current_start_pos = seg['start']
                    else:
                        # 段落本身就超大，按标点切分
                        sub_chunks = self._split_by_punctuation(seg_text, target_size)
                        for i, sub in enumerate(sub_chunks):
                            if i == 0:
                                current_parts = [sub]
                                current_length = len(sub)
                                current_start_pos = seg['start']
                            else:
                                flush()
                                overlap_text = self._get_smart_overlap_before(text, seg['start'], overlap, structures)
                                if overlap_text:
                                    current_parts = [overlap_text, sub]
                                else:
                                    current_parts = [sub]
                                _recompute_length()
                                current_start_pos = seg['start']
                            # 如果不是最后一个，立即 flush
                            if i < len(sub_chunks) - 1:
                                flush()

                elif projected_length <= capacity:
                    # 能放下
                    append_to_current(seg_text)

                else:
                    # 放不下：flush，新 chunk = 前向 overlap + 段落
                    if seg_len <= target_size:
                        start_new_chunk_with_overlap(seg, seg_text, is_anchor=False)
                    else:
                        # 段落本身超大，按标点切分
                        flush()
                        sub_chunks = self._split_by_punctuation(seg_text, target_size)
                        for i, sub in enumerate(sub_chunks):
                            overlap_text = self._get_smart_overlap_before(text, seg['start'], overlap, structures)
                            if overlap_text and i == 0:
                                current_parts = [overlap_text, sub]
                            else:
                                current_parts = [sub]
                            _recompute_length()
                            current_start_pos = seg['start']
                            if i < len(sub_chunks) - 1:
                                flush()

        # 保存最后一个 chunk
        flush()

        return chunks if chunks else [text]

    def _extract_structures(self, text: str) -> List[Dict]:
        """
        提取所有表格和图片的位置信息（最高优先级）

        Args:
            text: 输入文本

        Returns:
            按位置排序的结构列表 [{'type': 'TABLE'|'IMAGE', 'start': int, 'end': int, 'content': str}, ...]
        """
        structures = []

        # 使用预编译的正则表达式提取所有表格
        for match in RE_TABLE_BLOCK.finditer(text):
            structures.append({
                'type': 'TABLE',
                'start': match.start(),
                'end': match.end(),
                'content': match.group()
            })

        # 使用预编译的正则表达式提取所有图片
        for match in RE_IMAGE_BLOCK.finditer(text):
            structures.append({
                'type': 'IMAGE',
                'start': match.start(),
                'end': match.end(),
                'content': match.group()
            })

        # 按开始位置排序
        structures.sort(key=lambda x: x['start'])

        return structures

    def _split_large_table(self, table_text: str, chunk_size: int) -> List[str]:
        """
        切分超大表格，保留表头

        支持单行HTML格式（所有内容在一行）和多行HTML格式

        格式示例：
        <!-- TABLE_START id=table_7 -->

        <table><tr><td>列1</td><td>列2</td></tr><tr><td>数据1</td><td>数据2</td></tr></table>

        <!-- TABLE_END id=table_7 -->

        Args:
            table_text: 表格的markdown文本（包含 TABLE_START/END 标记）
            chunk_size: 目标块大小

        Returns:
            切分后的表格块列表
        """
        import re

        # 提取 TABLE_START 和 TABLE_END 标记
        table_start_match = re.search(r'<!--\s*TABLE_START[^>]*?-->', table_text)
        table_end_match = re.search(r'<!--\s*TABLE_END[^>]*?-->', table_text)

        if not table_start_match or not table_end_match:
            # 如果找不到标记，直接返回原文本
            return [table_text]

        table_start_marker = table_start_match.group()
        table_end_marker = table_end_match.group()

        # 提取 HTML 表格内容（支持单行或多行）
        html_match = re.search(r'<table>.*?</table>', table_text, re.DOTALL)
        if not html_match:
            # 如果找不到表格，直接返回原文本
            return [table_text]

        html_table = html_match.group()

        # 解析 HTML 表格，提取所有的 <tr>...</tr>（支持单行HTML）
        # re.DOTALL 确保 . 可以匹配换行符，适用于多行HTML
        tr_pattern = re.compile(r'<tr>(.*?)</tr>', re.DOTALL)
        rows = tr_pattern.findall(html_table)

        if not rows:
            # 如果没有行，直接返回原文本
            return [table_text]

        # 第一行作为表头
        header_row = f'<tr>{rows[0]}</tr>'
        data_rows = [f'<tr>{row}</tr>' for row in rows[1:]]

        if not data_rows:
            # 如果只有表头，直接返回原文本
            return [table_text]

        # 构建表头部分（包含 TABLE_START 标记和表头行）
        header_table = f'<table>{header_row}'
        header_part = f'{table_start_marker}\n\n{header_table}'
        header_size = len(header_part)

        # 计算每块能容纳多少数据行
        # 需要预留空间给结束标签 </table> 和 TABLE_END 标记
        footer_size = len('</table>') + len(f'\n\n{table_end_marker}')
        available_size = chunk_size - header_size - footer_size - 50  # 预留50字符缓冲

        if available_size <= 0:
            # 如果连基础结构都放不下，说明 chunk_size 设置太小，返回原文本
            return [table_text]

        sub_chunks = []
        current_chunk_rows = []
        current_size = 0

        for row in data_rows:
            row_size = len(row)

            # 特殊情况：单行就超过可用空间
            if row_size > available_size:
                # 先保存当前累积的行
                if current_chunk_rows:
                    chunk_body = ''.join(current_chunk_rows)
                    chunk = f'{header_part}{chunk_body}</table>\n\n{table_end_marker}'
                    sub_chunks.append(chunk)
                    current_chunk_rows = []
                    current_size = 0

                # 超长行单独成块（即使超过chunk_size）
                chunk_body = row
                chunk = f'{header_part}{chunk_body}</table>\n\n{table_end_marker}'
                sub_chunks.append(chunk)
                continue

            # 正常情况：检查是否放得下
            if current_size + row_size > available_size and current_chunk_rows:
                # 当前块满了，生成一个子块
                chunk_body = ''.join(current_chunk_rows)
                chunk = f'{header_part}{chunk_body}</table>\n\n{table_end_marker}'
                sub_chunks.append(chunk)

                # 重置
                current_chunk_rows = []
                current_size = 0

            current_chunk_rows.append(row)
            current_size += row_size

        # 处理最后一块
        if current_chunk_rows:
            chunk_body = ''.join(current_chunk_rows)
            chunk = f'{header_part}{chunk_body}</table>\n\n{table_end_marker}'
            sub_chunks.append(chunk)

        return sub_chunks if sub_chunks else [table_text]

    def _get_smart_overlap_before(self, text: str, start_pos: int, desired_length: int, structures: List[Dict]) -> str:
        """
        智能提取表格/图片之前的 overlap 文本，避免截断表格、图片、标题和句子

        策略：
        1. 从 start_pos 往前取 desired_length 长度的文本
        2. 检查这段文本是否包含结构边界（表格/图片的开始或结束）
        3. 如果包含，则在边界处截断
        4. 检查是否包含标题标记（#, ##, ###），如果有则在标题后截断
        5. 检查是否截断了句子，如果有则找到完整的句子边界

        Args:
            text: 完整文本
            start_pos: 当前位置
            desired_length: 期望的overlap长度
            structures: 结构列表

        Returns:
            智能截断后的overlap文本
        """
        # 计算实际的起始位置
        actual_start = max(0, start_pos - desired_length)
        overlap_text = text[actual_start:start_pos]

        if not overlap_text:
            return ""

        # 检查这段 overlap 是否与任何结构重叠
        for struct in structures:
            struct_start = struct['start']
            struct_end = struct['end']

            # 情况1: overlap 包含结构的结束部分（最常见的问题）
            if actual_start < struct_end <= start_pos:
                # 从结构结束位置开始取
                overlap_text = text[struct_end:start_pos]
                actual_start = struct_end

            # 情况2: overlap 包含结构的开始部分
            elif actual_start <= struct_start < start_pos:
                # 只取到结构开始位置
                overlap_text = text[actual_start:struct_start]
                break

            # 情况3: overlap 完全在结构内部
            elif struct_start <= actual_start and start_pos <= struct_end:
                # 不应该发生，但为了安全返回空
                return ""

        # 检查是否包含标题标记，找到最后一个标题的位置
        headers = list(RE_HEADER_PATTERN.finditer(overlap_text))

        if headers:
            # 从最后一个标题之后开始
            last_header = headers[-1]
            header_end_pos = last_header.start()
            overlap_text = overlap_text[header_end_pos:].lstrip('\n')

        # 智能处理句子边界，避免截断句子
        overlap_text = self._trim_to_sentence_boundary(overlap_text)

        return overlap_text.strip()

    def _trim_to_sentence_boundary(self, text: str) -> str:
        """
        将文本修剪到完整的句子边界或行边界，避免截断句子/行开头

        策略：
        1. 优先查找句子结束标记（。！？；\n\n）
        2. 如果没有句子标记，查找行边界（\n）
        3. 从边界之后开始返回文本
        4. 如果都没找到，返回原文本

        Args:
            text: 输入文本

        Returns:
            修剪后的文本
        """
        if not text:
            return text

        # 策略1：使用预编译的正则查找第一个句子结束标记
        match = RE_SENTENCE_END.search(text)

        if match:
            # 从第一个句子结束标记之后开始
            trim_pos = match.end()
            trimmed_text = text[trim_pos:].lstrip()

            # 如果修剪后的文本太短，尝试使用行边界
            if len(trimmed_text) < OVERLAP_MIN_LENGTH:
                # 尝试策略2：从第一个完整行开始
                first_newline = text.find('\n')
                if first_newline != -1:
                    trimmed_text = text[first_newline + 1:].lstrip()
                    if len(trimmed_text) >= OVERLAP_MIN_LENGTH:
                        return trimmed_text
                return ""

            return trimmed_text

        # 策略2：没有句子边界，尝试从完整的行边界开始（适用于目录、表格等无标点内容）
        first_newline = text.find('\n')
        if first_newline != -1:
            trimmed_text = text[first_newline + 1:].lstrip()
            if len(trimmed_text) >= OVERLAP_MIN_LENGTH:
                return trimmed_text

        # 策略3：没有找到任何边界，返回原文本
        return text

    def _get_smart_context_after(self, text: str, end_pos: int, desired_length: int, structures: List[Dict]) -> str:
        """
        智能提取表格/图片之后的上下文文本，避免截断表格、图片、标题和句子

        策略：
        1. 从 end_pos 往后取 desired_length 长度的文本
        2. 检查这段文本是否包含结构边界（表格/图片的开始或结束）
        3. 如果包含，则在边界前截断
        4. 检查是否截断了句子，如果有则找到完整的句子边界

        Args:
            text: 完整文本
            end_pos: 当前结束位置
            desired_length: 期望的上下文长度
            structures: 结构列表

        Returns:
            智能截断后的上下文文本
        """
        if end_pos >= len(text):
            return ""

        if desired_length <= 0:
            return ""

        # 计算要提取的结束位置
        extract_end = min(end_pos + desired_length, len(text))
        text_after = text[end_pos:extract_end]

        if not text_after.strip():
            return ""

        # 策略1：检查是否包含其他结构的边界
        for struct in structures:
            # 检查是否会包含其他结构的开始或结束
            if end_pos < struct['start'] < extract_end:
                # 在下一个结构开始前截断
                text_after = text[end_pos:struct['start']]
                break
            if end_pos < struct['end'] < extract_end:
                # 在结构结束后截断
                text_after = text[end_pos:struct['end']]
                break

        if not text_after.strip():
            return ""

        # 策略2：在完整的句子边界处截断
        # 找到最后一个句子结束符
        last_sentence_end = -1
        for match in RE_SENTENCE_END.finditer(text_after):
            last_sentence_end = match.end()

        if last_sentence_end > 0:
            # 在句子边界处截断
            trimmed_text = text_after[:last_sentence_end].rstrip()
            if len(trimmed_text) >= OVERLAP_MIN_LENGTH:
                return trimmed_text

        # 策略3：在完整的行边界处截断
        last_newline = text_after.rfind('\n')
        if last_newline != -1:
            trimmed_text = text_after[:last_newline].rstrip()
            if len(trimmed_text) >= OVERLAP_MIN_LENGTH:
                return trimmed_text

        # 策略4：返回原文
        return text_after.rstrip()

    def _chunk_plain_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        对纯文本（无表格/图片）按段落和标点符号切分，智能处理overlap避免截断句子

        Args:
            text: 纯文本
            chunk_size: 块大小
            overlap: 重叠大小

        Returns:
            文本块列表
        """
        chunks = []
        current_parts = []  # 使用列表累积
        current_length = 0  # 跟踪当前长度
        previous_text = ""

        def flush_current():
            """将当前累积的内容保存为一个 chunk"""
            nonlocal current_parts, current_length, previous_text
            if current_parts:
                chunk_text = ''.join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                    previous_text = self._get_smart_overlap_for_plain_text(chunk_text, overlap)
            current_parts = []
            current_length = 0

        def reset_current(text: str = ""):
            """重置当前累积，可选地设置初始内容"""
            nonlocal current_parts, current_length
            if text:
                current_parts = [text]
                current_length = len(text)
            else:
                current_parts = []
                current_length = 0

        paragraphs = text.split('\n\n')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 段落太长，按标点切分
            if len(para) > chunk_size:
                flush_current()

                sub_chunks = self._split_by_punctuation(para, chunk_size)
                for sub in sub_chunks:
                    sub_with_sep = sub + ' '
                    if current_length + len(sub_with_sep) < chunk_size:
                        current_parts.append(sub_with_sep)
                        current_length += len(sub_with_sep)
                    else:
                        flush_current()
                        reset_current(previous_text + sub + ' ')

            # 尝试累积段落
            else:
                para_with_sep = para + '\n\n'
                if current_length + len(para_with_sep) < chunk_size:
                    current_parts.append(para_with_sep)
                    current_length += len(para_with_sep)
                else:
                    flush_current()
                    reset_current(previous_text + para + '\n\n')

        flush_current()

        return chunks if chunks else [text]

    def _get_smart_overlap_for_plain_text(self, text: str, desired_length: int) -> str:
        """
        为纯文本获取智能overlap，避免截断句子

        Args:
            text: 当前chunk文本
            desired_length: 期望的overlap长度

        Returns:
            智能截断后的overlap文本
        """
        if len(text) <= desired_length:
            return text

        # 从后往前取desired_length长度
        overlap_text = text[-desired_length:]

        # 使用句子边界修剪
        trimmed = self._trim_to_sentence_boundary(overlap_text)

        return trimmed

    def _split_by_punctuation(self, text: str, max_length: int) -> List[str]:
        """
        按标点符号切分超长文本

        Args:
            text: 输入文本
            max_length: 最大长度

        Returns:
            切分后的文本列表
        """
        # 使用预编译的正则按中英文标点符号切分
        sentences = RE_PUNCTUATION_SPLIT.split(text)

        # 重新组合句子和标点
        combined = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                combined.append(sentences[i] + sentences[i+1])
            else:
                combined.append(sentences[i])
        if len(sentences) % 2 == 1:
            combined.append(sentences[-1])

        # 如果单个句子仍然太长，强制按字符切分
        result = []
        for sentence in combined:
            if len(sentence) > max_length:
                # 强制切分
                for i in range(0, len(sentence), max_length):
                    result.append(sentence[i:i+max_length])
            else:
                result.append(sentence)

        return result

    def _clean_content_for_embedding(self, text: str) -> str:
        """
        清洗文本内容，移除 HTML 注释标记和图片 markdown 标记，用于向量化

        移除内容：
        1. TABLE_START/END 注释：<!-- TABLE_START id=... --> 和 <!-- TABLE_END -->
        2. IMAGE_START/END 注释：<!-- IMAGE_START id=... --> 和 <!-- IMAGE_END -->
        3. 图片 markdown 语法：![图片](images/...)

        Args:
            text: 输入文本

        Returns:
            清洗后的文本
        """
        # 使用预编译的正则表达式移除各类标记
        text = RE_TABLE_START.sub('', text)
        text = RE_TABLE_END.sub('', text)
        text = RE_IMAGE_START.sub('', text)
        text = RE_IMAGE_END.sub('', text)
        text = RE_IMAGE_MARKDOWN.sub('', text)

        # 清理多余的空行（连续3个以上换行符压缩为2个）
        text = RE_MULTIPLE_NEWLINES.sub('\n\n', text)

        return text.strip()
    
    def process_markdown_file(self, file_path: Path, chunk_size: int = 2000, overlap: int = 200, encode_batch_size: int = 32) -> List[Dict]:
        """
        处理单个markdown文件

        Args:
            file_path: markdown文件路径
            chunk_size: 目标块大小（字符数）
            overlap: 重叠大小
            encode_batch_size: 向量化批次大小（CPU建议32-48，GPU建议64-128）

        Returns:
            处理后的数据列表
        """
        # 从文件名提取基金代码和名称
        filename = file_path.stem
        parts = filename.split('_')
        fund_code = parts[0] if parts else "unknown"
        fund_name = '_'.join(parts[1:]).replace('_analyzed', '') if len(parts) > 1 else "unknown"

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用新的多阶段分块策略
        chunk_dicts = self.chunk_text_with_headers(content, chunk_size=chunk_size, overlap=overlap)

        # 准备向量化内容：对 content 进行清洗，移除 HTML 注释和图片标记
        chunks_content_for_embedding = [
            self._clean_content_for_embedding(chunk_dict['content'])
            for chunk_dict in chunk_dicts
        ]

        # 分批生成向量（稠密+稀疏），避免内存溢出
        print(f"  处理 {len(chunks_content_for_embedding)} 个文本块（批次大小: {encode_batch_size}）...")
        all_dense_embeddings = []
        all_sparse_embeddings = []
        for i in range(0, len(chunks_content_for_embedding), encode_batch_size):
            batch_content = chunks_content_for_embedding[i:i+encode_batch_size]
            # BGE-M3同时生成稠密和稀疏向量
            batch_output = self.model.encode(batch_content, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            all_dense_embeddings.extend(batch_output['dense_vecs'])
            all_sparse_embeddings.extend(batch_output['lexical_weights'])
            print(f"    已完成 {min(i+encode_batch_size, len(chunks_content_for_embedding))}/{len(chunks_content_for_embedding)} 个块")

        # 准备数据
        data_list = []
        for i, (chunk_dict, dense_emb, sparse_emb) in enumerate(zip(chunk_dicts, all_dense_embeddings, all_sparse_embeddings)):
            # 生成唯一ID
            unique_str = f"{fund_code}_{i}_{chunk_dict['content'][:50]}"
            doc_id = hashlib.md5(unique_str.encode()).hexdigest()[:32]

            # 确保 content 长度不超过 schema 限制
            # 注意：现在 content 已经包含了标题路径前缀，可能会更长
            original_content = chunk_dict['content']
            if len(original_content) > CONTENT_MAX_LENGTH:
                # 如果超长，截断并添加省略提示
                truncated_content = original_content[:CONTENT_MAX_LENGTH - 50] + "\n\n[内容因长度限制被截断...]"
            else:
                truncated_content = original_content

            # 构建数据字典
            data_item = {
                "id": doc_id,
                "fund_code": fund_code,
                "fund_name": fund_name,
                "file_path": str(file_path),
                "content": truncated_content,  # 存储原始内容（包含标题路径），但不超过限制
                "chunk_index": i,
                "embedding": dense_emb.tolist(),
                "sparse_embedding": sparse_emb  # 稀疏向量已经是字典格式 {token_id: weight}
            }

            # 添加 header 字段
            for j in range(1, MAX_HEADER_LEVELS + 1):
                header_value = chunk_dict.get(f'header_{j}', '')
                data_item[f'header_{j}'] = header_value[:500] if header_value else ""

            data_list.append(data_item)

        return data_list

    def batch_insert(self, data_list: List[Dict], batch_size: int = 100):
        """
        批量插入数据到Milvus

        Args:
            data_list: 数据列表
            batch_size: 批次大小（CPU建议100-200，GPU建议500+）
        """
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]

            # 使用新的MilvusClient API插入数据
            self.client.insert(
                collection_name=self.collection_name,
                data=batch
            )
            print(f"  已插入 {min(i+batch_size, len(data_list))}/{len(data_list)} 条数据")

        print(f"完成插入 {len(data_list)} 条数据")

    def get_indexed_fund_codes(self) -> set[str]:
        """查询 Milvus 中已入库的基金代码集合。"""
        if not self.client.has_collection(self.collection_name):
            return set()
        results = self.client.query(
            collection_name=self.collection_name,
            filter="fund_code != ''",
            output_fields=["fund_code"],
            limit=10000,
        )
        return {r["fund_code"] for r in results if r.get("fund_code")}

    def delete_fund(self, fund_code: str) -> int:
        """删除指定基金的所有 chunk 数据。

        Args:
            fund_code: 基金代码，如 "159103"

        Returns:
            删除的记录数
        """
        if not self.client.has_collection(self.collection_name):
            return 0

        # 使用 query_iterator 分页查询，自动处理 Milvus 的 max_query_result_window 限制
        all_results = []
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            filter=f'fund_code == "{fund_code}"',
            output_fields=["id"],
            batch_size=1000,
        )
        while True:
            batch = iterator.next()
            if not batch:
                break
            all_results.extend(batch)
        iterator.close()

        if not all_results:
            print(f"  基金 {fund_code} 在集合中无数据")
            return 0

        ids = [r["id"] for r in all_results]
        self.client.delete(
            collection_name=self.collection_name,
            ids=ids,
        )
        print(f"  已删除基金 {fund_code} 的 {len(ids)} 条 chunk 数据")
        return len(ids)

    def process_all_files(
        self,
        markdown_dir: str,
        chunk_size: int = 2000,
        overlap: int = 200,
        encode_batch_size: int = 32,
        insert_batch_size: int = 100,
        accumulate_threshold: int = 500,
        force_codes: list[str] | None = None,
    ):
        """处理所有analyzed markdown文件（增量模式）。

        默认只处理 Milvus 中尚未入库的基金，已有基金自动跳过。

        Args:
            markdown_dir:        markdown输出目录
            chunk_size:          目标块大小（字符数）
            overlap:             重叠大小
            encode_batch_size:   向量化批次大小（CPU建议32-48）
            insert_batch_size:   Milvus插入批次大小（CPU建议100-200）
            accumulate_threshold: 累积多少条数据后插入一次（建议500-1000）
            force_codes:         强制重新处理的基金代码列表（会先删除旧数据再插入）。
                                 None 表示不强制覆盖任何基金。
                                 传入 ["ALL"] 等价于全量重建（慎用）。
        """
        markdown_path = Path(markdown_dir)

        # 查找所有analyzed文件
        analyzed_files = list(markdown_path.glob("**/*_analyzed.md"))
        print(f"找到 {len(analyzed_files)} 个analyzed文件")

        if not analyzed_files:
            print("未找到任何analyzed文件，请先运行markdown分析")
            return

        # ── 查询已入库的基金代码 ──
        indexed_codes = self.get_indexed_fund_codes()
        print(f"Milvus 中已有 {len(indexed_codes)} 只基金")

        force_set: set[str] = set(force_codes) if force_codes else set()
        force_all = "ALL" in force_set

        # ── 分类文件：跳过 / 强制覆盖 / 新增 ──
        to_process: list[tuple[Path, str, bool]] = []  # (path, fund_code, need_delete_first)
        skipped = 0

        for file_path in analyzed_files:
            parts = file_path.stem.split('_')
            fund_code = parts[0] if parts else "unknown"

            if force_all or fund_code in force_set:
                to_process.append((file_path, fund_code, True))   # 强制覆盖：先删后插
            elif fund_code in indexed_codes:
                skipped += 1                                       # 已有，跳过
            else:
                to_process.append((file_path, fund_code, False))  # 新增：直接插入

        print(f"  跳过（已入库）: {skipped} 只")
        print(f"  待处理（新增）: {sum(1 for _, _, d in to_process if not d)} 只")
        print(f"  待处理（强制覆盖）: {sum(1 for _, _, d in to_process if d)} 只")

        if not to_process:
            print("无需处理，所有基金已入库。如需强制覆盖，请传入 force_codes 参数。")
            return

        print(f"\n配置参数:")
        print(f"  分块大小: {chunk_size}")
        print(f"  重叠大小: {overlap}")
        print(f"  向量化批次大小: {encode_batch_size}")
        print(f"  插入批次大小: {insert_batch_size}")
        print(f"  累积阈值: {accumulate_threshold}")
        print()

        # ── 处理文件 ──
        all_data = []
        for file_path, fund_code, need_delete in tqdm(to_process, desc="处理文件"):
            try:
                print(f"\n处理文件: {file_path.name}{'（覆盖）' if need_delete else '（新增）'}")

                # 强制覆盖：先删除该基金的旧数据
                if need_delete:
                    self.delete_fund(fund_code)

                data_list = self.process_markdown_file(
                    file_path,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    encode_batch_size=encode_batch_size,
                )
                all_data.extend(data_list)

                # 累积到阈值后批量插入
                if len(all_data) >= accumulate_threshold:
                    print(f"\n累积 {len(all_data)} 条数据，开始插入...")
                    self.batch_insert(all_data, batch_size=insert_batch_size)
                    all_data = []

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 插入剩余数据
        if all_data:
            print(f"\n插入剩余 {len(all_data)} 条数据...")
            self.batch_insert(all_data, batch_size=insert_batch_size)

    def search(self, query: str, top_k: int = 5, search_type: str = "hybrid") -> List[Dict]:
        """
        搜索相似内容

        Args:
            query: 查询文本
            top_k: 返回top k结果
            search_type: 搜索类型 - "dense"(仅稠密), "sparse"(仅神经稀疏), "hybrid"(混合RRF)

        Returns:
            搜索结果列表
        """
        if search_type == "dense":
            # 仅稠密向量检索
            return self._search_dense(query, top_k)
        elif search_type == "sparse":
            # 仅稀疏向量检索
            return self._search_sparse(query, top_k)
        elif search_type == "hybrid":
            # 混合检索 + RRF融合
            return self._search_hybrid(query, top_k)
        else:
            raise ValueError(f"不支持的搜索类型: {search_type}")

    def _search_dense(self, query: str, top_k: int = 5) -> List[Dict]:
        """仅使用稠密向量检索"""
        # 生成查询向量
        query_embedding = self.model.encode([query], return_dense=True, return_sparse=False)['dense_vecs'][0].tolist()

        # 搜索
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            limit=top_k,
            output_fields=["fund_code", "fund_name", "content", "chunk_index", "header_1", "header_2", "header_3", "header_4", "header_5", "header_6"]
        )

        return self._format_results(results)

    def _search_sparse(self, query: str, top_k: int = 5) -> List[Dict]:
        """仅使用神经稀疏向量检索 (Learned Sparse Retrieval)"""
        # 生成稀疏向量
        query_sparse = self.model.encode([query], return_dense=False, return_sparse=True)['lexical_weights'][0]

        # 搜索
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_sparse],
            anns_field="sparse_embedding",
            limit=top_k,
            output_fields=["fund_code", "fund_name", "content", "chunk_index", "header_1", "header_2", "header_3", "header_4", "header_5", "header_6"]
        )

        return self._format_results(results)

    def _search_hybrid(self, query: str, top_k: int = 5) -> List[Dict]:
        """混合检索：稠密+稀疏 RRF融合"""
        from hybrid_search import rrf_fusion, deduplicate_results

        # 获取2倍的结果用于融合
        dense_results = self._search_dense(query, top_k * 2)
        sparse_results = self._search_sparse(query, top_k * 2)

        # RRF融合
        fused_results = rrf_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=60,
            dense_weight=0.5,
            sparse_weight=0.5
        )

        # 去重
        dedup_results = deduplicate_results(fused_results)

        # 返回top_k
        return dedup_results[:top_k]

    def _format_results(self, results) -> List[Dict]:
        """格式化Milvus搜索结果"""
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "fund_code": hit.get("entity", {}).get("fund_code", hit.get("fund_code")),
                    "fund_name": hit.get("entity", {}).get("fund_name", hit.get("fund_name")),
                    "content": hit.get("entity", {}).get("content", hit.get("content")),
                    "chunk_index": hit.get("entity", {}).get("chunk_index", hit.get("chunk_index")),
                    "header_1": hit.get("entity", {}).get("header_1", hit.get("header_1", "")),
                    "header_2": hit.get("entity", {}).get("header_2", hit.get("header_2", "")),
                    "header_3": hit.get("entity", {}).get("header_3", hit.get("header_3", "")),
                    "header_4": hit.get("entity", {}).get("header_4", hit.get("header_4", "")),
                    "header_5": hit.get("entity", {}).get("header_5", hit.get("header_5", "")),
                    "header_6": hit.get("entity", {}).get("header_6", hit.get("header_6", "")),
                    "score": hit.get("distance", 0)
                })
        return formatted_results


class FundIndexBuilder:
    """基金识别索引构建器（两级RAG的第一级）

    将每只基金的代码、名称、别名等标识信息向量化，
    存入独立的轻量 collection（fund_index）。
    查询时通过语义检索识别用户问题中提到的基金，
    替代原有基于字符串匹配的 FundRegistry。

    索引文本格式（向量化的内容）：
        "{code} {full_name} {short_name} {etf_short}"
    例：
        "159103 汇添富中证金融科技主题交易型开放式指数证券投资基金 金融科技ETF汇添富 金融科技ETF"
    """

    FUND_INDEX_COLLECTION = "fund_index"
    # fund_index 的 content 较短，500字符足够
    FUND_INDEX_MAX_CONTENT = 500

    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: int = 19595,
        model_path: str = "./embedding_model/bge-m3",
    ):
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        print(f"加载BGE-M3模型: {model_path}")
        self.model = BGEM3FlagModel(model_path, use_fp16=True)

        print(f"连接Milvus: {milvus_host}:{milvus_port}")
        self.client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")

    # ------------------------------------------------------------------
    # Collection 管理
    # ------------------------------------------------------------------

    def _setup_fund_index_collection(self, recreate: bool = False) -> None:
        """创建或重建 fund_index collection。

        Schema：
          id            VARCHAR(32)   主键，md5(fund_code)
          fund_code     VARCHAR(10)   基金代码，如 "159103"
          full_name     VARCHAR(500)  完整基金名称
          short_name    VARCHAR(200)  简短名称（ETF短名）
          index_text    VARCHAR(500)  向量化的拼接文本（用于检索）
          embedding     FLOAT_VECTOR(1024)  稠密向量
          sparse_embedding SPARSE_FLOAT_VECTOR 神经稀疏向量
        """
        col = self.FUND_INDEX_COLLECTION

        if self.client.has_collection(col):
            if recreate:
                self.client.drop_collection(col)
                print(f"已删除旧 collection: {col}")
            else:
                print(f"collection {col} 已存在，跳过创建")
                return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id",               DataType.VARCHAR, is_primary=True, max_length=32)
        schema.add_field("fund_code",        DataType.VARCHAR, max_length=10)
        schema.add_field("full_name",        DataType.VARCHAR, max_length=500)
        schema.add_field("short_name",       DataType.VARCHAR, max_length=200)
        schema.add_field("index_text",       DataType.VARCHAR, max_length=self.FUND_INDEX_MAX_CONTENT)
        schema.add_field("embedding",        DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )

        self.client.create_collection(
            collection_name=col,
            schema=schema,
            index_params=index_params,
        )
        print(f"✓ collection {col} 创建完成")

    # ------------------------------------------------------------------
    # 从 fund_reports collection 中读取基金清单
    # ------------------------------------------------------------------

    def _load_fund_list_from_reports(
        self,
        reports_collection: str = "fund_reports_mineru",
    ) -> List[Dict]:
        """从已有的年报 collection 中提取去重后的基金清单。

        Returns:
            [{"fund_code": "159103", "fund_name": "金融科技ETF汇添富"}, ...]
        """
        results = self.client.query(
            collection_name=reports_collection,
            filter="fund_code != ''",
            output_fields=["fund_code", "fund_name"],
            limit=10000,
        )

        seen: dict[str, str] = {}
        for r in results:
            code = r.get("fund_code", "").strip()
            name = r.get("fund_name", "").strip()
            if code and code not in seen:
                seen[code] = name

        funds = [{"fund_code": k, "fund_name": v} for k, v in sorted(seen.items())]
        print(f"从 {reports_collection} 中读取到 {len(funds)} 只基金")
        return funds

    # ------------------------------------------------------------------
    # 索引文本构造
    # ------------------------------------------------------------------

    @staticmethod
    def _build_index_text(code: str, full_name: str, short_name: str) -> str:
        """拼接向量化文本。

        格式："{code} {full_name} {short_name} {etf_short}"
        其中 etf_short 是去掉管理公司后缀的纯产品名，
        例如 "金融科技ETF汇添富" → "金融科技ETF"。

        多种表述拼在同一个句子里，BGE-M3 编码后能覆盖用户的各种叫法。
        """
        # 尝试提取纯 ETF 名（去掉最后的公司名后缀）
        # 规律：short_name 通常形如 "金融科技ETF汇添富"，公司名在最后
        etf_pure = short_name
        # 常见管理公司名（用于剥离后缀）
        company_suffixes = [
            "华夏", "华安", "汇添富", "易方达", "南方", "万家", "招商", "广发",
            "天弘", "博时", "鹏华", "嘉实", "大成", "工银瑞信", "工银", "永赢",
            "景顺长城", "景顺", "浦银安盛", "融通", "华泰柏瑞", "国联安", "浦银",
            "银华", "华宝", "国泰"
        ]
        for suffix in sorted(company_suffixes, key=len, reverse=True):
            if short_name.endswith(suffix) and len(short_name) > len(suffix):
                etf_pure = short_name[: -len(suffix)]
                break

        parts = [code, full_name, short_name]
        if etf_pure != short_name:
            parts.append(etf_pure)

        text = " ".join(parts)
        return text[:500]  # 不超过字段限制

    # ------------------------------------------------------------------
    # 构建索引主入口
    # ------------------------------------------------------------------

    def build_fund_index(
        self,
        reports_collection: str = "fund_reports_mineru",
        recreate: bool = False,
        encode_batch_size: int = 32,
    ) -> int:
        """构建/增量更新基金识别索引。

        增量模式（recreate=False，默认）：
          - 从年报 collection 读取所有基金
          - 与 fund_index 中已有记录对比
          - 只对新基金进行向量化并追加写入

        全量重建（recreate=True）：
          - 删除并重建 fund_index collection
          - 对所有基金重新向量化并写入

        Args:
            reports_collection: 从哪个年报 collection 读取基金清单
            recreate:           True = 强制全量重建
            encode_batch_size:  向量化批次大小

        Returns:
            本次新写入的基金数量
        """
        # 1. 建 collection（recreate=True 时先删再建）
        self._setup_fund_index_collection(recreate=recreate)

        # 2. 读取年报 collection 中的全量基金清单
        all_funds = self._load_fund_list_from_reports(reports_collection)
        if not all_funds:
            print("未读取到任何基金，中止")
            return 0

        # 3. 增量对比：找出 fund_index 中尚未入库的基金
        if not recreate and self.client.has_collection(self.FUND_INDEX_COLLECTION):
            existing = self.client.query(
                collection_name=self.FUND_INDEX_COLLECTION,
                filter="fund_code != ''",
                output_fields=["fund_code"],
                limit=10000,
            )
            existing_codes = {r["fund_code"] for r in existing}
            new_funds = [f for f in all_funds if f["fund_code"] not in existing_codes]
            print(f"fund_index 已有 {len(existing_codes)} 只，本次新增 {len(new_funds)} 只")
        else:
            new_funds = all_funds
            print(f"全量构建，共 {len(new_funds)} 只基金")

        if not new_funds:
            print("fund_index 已是最新，无需更新")
            return 0

        # 4. 构造索引文本
        records = []
        for f in new_funds:
            code      = f["fund_code"]
            full_name = f["fund_name"]
            short_name = full_name.split("_")[0] if "_" in full_name else full_name
            index_text = self._build_index_text(code, full_name, short_name)
            records.append({
                "fund_code":  code,
                "full_name":  full_name[:500],
                "short_name": short_name[:200],
                "index_text": index_text,
            })

        # 5. 批量向量化
        print(f"开始向量化 {len(records)} 条基金索引记录（批次: {encode_batch_size}）...")
        texts = [r["index_text"] for r in records]

        all_dense, all_sparse = [], []
        for i in range(0, len(texts), encode_batch_size):
            batch = texts[i : i + encode_batch_size]
            out = self.model.encode(
                batch,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            all_dense.extend(out["dense_vecs"])
            all_sparse.extend(out["lexical_weights"])
            print(f"  已向量化 {min(i + encode_batch_size, len(texts))}/{len(texts)}")

        # 6. 组装数据并插入
        import hashlib
        data_list = []
        for record, dense_emb, sparse_emb in zip(records, all_dense, all_sparse):
            doc_id = hashlib.md5(record["fund_code"].encode()).hexdigest()[:32]
            data_list.append({
                "id":               doc_id,
                "fund_code":        record["fund_code"],
                "full_name":        record["full_name"],
                "short_name":       record["short_name"],
                "index_text":       record["index_text"],
                "embedding":        dense_emb.tolist(),
                "sparse_embedding": sparse_emb,
            })

        self.client.insert(
            collection_name=self.FUND_INDEX_COLLECTION,
            data=data_list,
        )
        print(f"✓ fund_index 更新完成，本次写入 {len(data_list)} 条记录")
        return len(data_list)


def main():
    """主函数"""
    # 配置（GPU电脑上运行时使用）
    MARKDOWN_OUTPUT_DIR = str(Path(__file__).parent.parent / "markdown_mineru")  # 基金报告Markdown文件目录
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")  # GPU电脑本地Milvus
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19595"))
    COLLECTION_NAME = "fund_reports_mineru"
    MODEL_PATH = str(Path(__file__).parent.parent / "embedding_model" / "bge-m3")

    # 分块参数
    CHUNK_SIZE = 2000  # 目标块大小
    OVERLAP = 200  # 重叠大小

    # 自动检测设备
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到设备: {device}")

    # 根据设备自动调整参数
    if device == "cuda":
        ENCODE_BATCH_SIZE = 64  # GPU默认值，可根据显存调整到128
        INSERT_BATCH_SIZE = 500
        ACCUMULATE_THRESHOLD = 1000
        print("使用GPU优化参数")
    else:
        ENCODE_BATCH_SIZE = 32
        INSERT_BATCH_SIZE = 100
        ACCUMULATE_THRESHOLD = 500
        print("使用CPU参数")

    # ── 第一步：向量化年报内容 ──
    vectorizer = FundVectorizer(
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        collection_name=COLLECTION_NAME,
        model_path=MODEL_PATH
    )

    vectorizer.process_all_files(
        MARKDOWN_OUTPUT_DIR,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        encode_batch_size=ENCODE_BATCH_SIZE,
        insert_batch_size=INSERT_BATCH_SIZE,
        accumulate_threshold=ACCUMULATE_THRESHOLD,
        force_codes=["160323","160314","159663","180202"]
    )

    # ── 第二步：构建基金识别索引（两级RAG第一级）──
    print("\n" + "=" * 50)
    print("开始构建基金识别索引（fund_index）...")
    print("=" * 50)

    index_builder = FundIndexBuilder(
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        model_path=MODEL_PATH,
    )
    index_builder.build_fund_index(
        reports_collection=COLLECTION_NAME,
        recreate=False,          # 已存在则跳过，避免重复构建
        encode_batch_size=ENCODE_BATCH_SIZE,
    )
    print("✓ 全部完成")


if __name__ == "__main__":
    main()
