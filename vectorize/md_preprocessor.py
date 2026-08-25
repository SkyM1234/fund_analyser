"""
MD预处理脚本 - 读取原始_raw.md文件，修复格式问题后再交由后续分析处理。

处理链:
  raw MD (MinerU输出) → md_preprocessor → 清洗后MD → markdown_image_table_analyze
"""
import re
import sys
from pathlib import Path
from typing import Callable


# ============================================================
# 格式修复规则
# ============================================================

def fix_main_title(content: str) -> str:
    """
    修复主标题格式问题：
      A. 日期被拼在 h1 标题末尾 → 分离日期到正文
         e.g. "# xxx基金2025 年年度报告2025 年 12 月 31 日"
              → "# xxx基金2025 年年度报告" + 空行 + "2025 年 12 月 31 日"
      B. "年年度报告" 是独立 h1 → 合并到前一个 h1 标题
         e.g. "# xxx基金" + 空行 + "# 2025 年年度报告" + 空行 + "日期"
              → "# xxx基金2025 年年度报告" + 空行 + "日期"
      D. "年年度报告" 是 h1 后的纯文本 → 合并到前一个 h1 标题
         e.g. "# xxx基金" + 空行 + "2025 年年度报告" + 空行 + "日期"
              → "# xxx基金2025 年年度报告" + 空行 + "日期"
      E. h1 被换行截断，续文在下一行 → 合并续文
         e.g. "# xxx证券投资" + 空行 + "基金（QDII）" + 空行 + "2025 年年度报告"
              → "# xxx证券投资基金（QDII）2025 年年度报告" + 空行 + "日期"
      F. 第一个 h1 缺失（只有纯文本基金名）→ 补 "# " 前缀
         e.g. "华夏磐晟灵活配置混合型" + 空行 + "证券投资基金（LOF）" + …
              → "# 华夏磐晟灵活配置混合型证券投资基金（LOF）2025 年年度报告" + …
    """
    lines = content.split('\n')
    date_re = r'(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)'

    # ---------- 找到第一个 h1（跳过 § 开头的章节标题） ----------
    h1_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^# [^#]', line) and '§' not in line:
            h1_idx = i
            break

    # ---------- 子修复 F: 第一个 h1 缺失，尝试从纯文本创建 ----------
    if h1_idx is None:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # 判断是否是基金名（包含基金/证券等关键词）
                if re.search(r'(基金|证券|指数|交易型|混合|债券|股票)', stripped):
                    lines[i] = '# ' + stripped
                    h1_idx = i
                    break
        if h1_idx is None:
            return content

    # ---------- 子修复 A: 日期拼在 h1 标题末尾 ----------
    h1_line = lines[h1_idx]
    date_at_end = re.search(date_re + r'\s*$', h1_line)
    if date_at_end:
        date_str = date_at_end.group(1)
        lines[h1_idx] = h1_line[:date_at_end.start()].rstrip()
        lines.insert(h1_idx + 1, '')
        lines.insert(h1_idx + 2, date_str)

    # ---------- 子修复 B/D/E: 扫描 h1 后的行，合并"年年度报告"和续文 ----------
    pos = h1_idx + 1
    while pos < len(lines) and pos < h1_idx + 12:
        line = lines[pos]
        stripped = line.strip()
        pos += 1  # 先推进索引；pop 操作时用 pos-1 定位

        if not stripped:
            continue  # 跳过空行

        # 遇到下一个标题（非年年度报告）→ 停止扫描
        if stripped.startswith('#'):
            # Fix B: h1 格式的年年度报告
            if re.match(r'^# \d{4}\s*年年度报告\s*$', stripped):
                report_part = stripped.lstrip('#').strip()
                lines[h1_idx] = lines[h1_idx] + report_part
                idx = pos - 1
                lines.pop(idx)
                if idx - 1 > h1_idx and lines[idx - 1].strip() == '':
                    lines.pop(idx - 1)
                pos = h1_idx + 1  # 回到 h1 后继续扫描
                continue
            # Fix G: 日期被 MinerU 错误识别为 h1 标题 → 降级为纯文本
            # 例如 "# 2025 年 12 月 31 日" → "2025 年 12 月 31 日"
            if re.match(r'^#\s*' + date_re + r'\s*$', stripped):
                idx = pos - 1
                lines[idx] = stripped.lstrip('#').strip()
                pos = h1_idx + 1  # 回到 h1 后继续扫描
                continue
            break  # 其他标题，停止

        # 日期行 → 已在正确位置，停止扫描
        if re.match(date_re + r'\s*$', stripped):
            break

        # 元数据行（基金管理人/托管人/送出日期）→ 停止扫描
        if re.match(r'^(基金管理人|基金托管人|送出日期)', stripped):
            break

        # Fix D: 纯文本格式的年年度报告
        if re.match(r'^\d{4}\s*年年度报告\s*$', stripped):
            lines[h1_idx] = lines[h1_idx] + stripped
            idx = pos - 1
            lines.pop(idx)
            if idx - 1 > h1_idx and lines[idx - 1].strip() == '':
                lines.pop(idx - 1)
            pos = h1_idx + 1
            continue

        # Fix E: 续文检测 — h1 被换行截断后的续行
        # 如 "基金（QDII）"、"证券投资基金（LOF）" 等
        if re.match(r'^(基金|证券|指数|交易型|混合|债券|股票|[（(])', stripped):
            lines[h1_idx] = lines[h1_idx] + stripped
            idx = pos - 1
            lines.pop(idx)
            if idx - 1 > h1_idx and lines[idx - 1].strip() == '':
                lines.pop(idx - 1)
            pos = h1_idx + 1
            continue

        # 无法识别的行 → 停止扫描
        break

    # ---------- 清理多余空行 ----------
    result = '\n'.join(lines)
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    return result


def fix_missing_toc_header(content: str) -> str:
    """
    补全缺失的 "1.2 目录" 标题。

    检测逻辑：
      1. 如果已存在 "#... 1.2 目录" 标题 → 跳过
      2. 找到 "1.1 重要提示" 的标题级别（# 数量）
      3. 在 1.1 内容之后、TOC（以 § 开头的行）之前插入同级标题
    """
    # 已存在则跳过
    if re.search(r'^#+\s+1\.2\s+目录', content, re.MULTILINE):
        return content

    # 找到 "1.1 重要提示" 获取标题级别
    m_1_1 = re.search(r'^(#+)\s+1\.1\s+重要提示', content, re.MULTILINE)
    if not m_1_1:
        return content  # 连 1.1 都没有，无法判断

    heading_prefix = m_1_1.group(1)  # e.g. "###"
    after_1_1_pos = m_1_1.end()

    # 在 1.1 之后的内容中，找到第一个 TOC 条目（以 § 开头、非标题行）
    remaining = content[after_1_1_pos:]
    m_toc = re.search(r'^§', remaining, re.MULTILINE)
    if not m_toc:
        return content  # 没找到 TOC

    insert_pos = after_1_1_pos + m_toc.start()

    # 在插入位置前确保有空行分隔
    header_line = f"{heading_prefix} 1.2 目录"
    content = content[:insert_pos] + header_line + '\n\n' + content[insert_pos:]

    return content


def fix_image_blank_lines(content: str) -> str:
    """
    确保图片引用（![](images/...)）上下有空行分隔。

    修复前:
      文字行
      ![](images/xxx.jpg)
      文字行

    修复后:
      文字行

      ![](images/xxx.jpg)

      文字行
    """
    # 匹配 MinerU 图片行: ![...](images/...) 可带尾随空格
    img_re = r'!\[[^\]]*\]\(images/[^)]+\)[ \t]*'

    # 图片前缺空行：非空行直接后接图片行 → 中间插入空行
    content = re.sub(
        rf'([^\n])\n({img_re})',
        r'\1\n\n\2',
        content,
    )

    # 图片后缺空行：图片行直接后接非空行 → 中间插入空行
    content = re.sub(
        rf'({img_re})\n([^\n])',
        r'\1\n\n\2',
        content,
    )

    return content


# ============================================================
# 预处理器
# ============================================================

class MarkdownPreprocessor:
    """Markdown预处理处理器 —— 对原始MD文件进行格式修复"""

    def __init__(self, append_mode: bool = False):
        """
        Args:
            append_mode: 追加模式，为 True 时跳过输出文件已存在的文件，只处理新文件
        """
        self.append_mode = append_mode
        # 格式修复规则列表: [(规则名称, 处理函数)]
        # 每个函数签名: (content: str) -> str
        self._fixes: list[tuple[str, Callable[[str], str]]] = []

    def register_fix(self, name: str, fix_func: Callable[[str], str]):
        """
        注册一个格式修复规则

        Args:
            name: 规则名称（用于日志输出）
            fix_func: 处理函数，接收原始内容，返回修复后的内容
        """
        self._fixes.append((name, fix_func))

    def apply_fixes(self, content: str) -> str:
        """
        按注册顺序依次应用所有修复规则

        Args:
            content: 原始MD内容

        Returns:
            修复后的MD内容
        """
        result = content
        for name, fix_func in self._fixes:
            print(f"    [预处理] {name}...")
            result = fix_func(result)
        return result

    def process_markdown_file(
        self,
        md_path: str,
        output_path: str = None,
    ) -> str:
        """
        处理单个MD文件

        Args:
            md_path: 原始MD文件路径（通常为 *_raw.md）
            output_path: 输出路径（不指定则在同目录生成 *_preprocessed.md）

        Returns:
            输出文件路径
        """
        md_path = Path(md_path)

        if output_path is None:
            # 生成 _preprocessed 结尾的文件名
            stem = md_path.stem
            if stem.endswith("_raw"):
                stem = stem[:-4]
            output_path = md_path.parent / f"{stem}_preprocessed.md"
        else:
            output_path = Path(output_path)

        # 追加模式：输出文件已存在则跳过
        if self.append_mode and output_path.exists():
            print(f"  [SKIP] 输出文件已存在，追加模式跳过: {output_path.name}")
            return str(output_path)

        print(f"\n{'='*60}")
        print(f"预处理文件: {md_path.name}")
        print(f"{'='*60}")

        # 读取原始内容
        with open(md_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # 应用所有修复规则
        if not self._fixes:
            print("  [WARN] 未注册任何修复规则，内容保持不变")
            processed_content = original_content
        else:
            print(f"  已注册 {len(self._fixes)} 条修复规则")
            processed_content = self.apply_fixes(original_content)

        # 写入
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)

        print(f"  [OK] 已保存: {output_path}")
        return str(output_path)

    def process_directory(
        self,
        input_dir: str,
        pattern: str = "*_raw.md",
        recursive: bool = True,
    ) -> list[str]:
        """
        批量处理目录下的MD文件

        Args:
            input_dir: MD文件所在目录
            pattern: 文件名匹配模式
            recursive: 是否递归处理子目录

        Returns:
            已处理的文件路径列表
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            print(f"错误: 目录不存在 - {input_dir}")
            return []

        if recursive:
            md_files = list(input_dir.rglob(pattern))
        else:
            md_files = list(input_dir.glob(pattern))

        if not md_files:
            print(f"在 {input_dir} 中未找到匹配 '{pattern}' 的MD文件")
            return []

        print(f"\n{'='*60}")
        print(f"批量预处理")
        print(f"{'='*60}")
        print(f"目录: {input_dir}")
        print(f"找到 {len(md_files)} 个MD文件")
        print(f"已注册 {len(self._fixes)} 条修复规则")
        print(f"追加模式: {'启用' if self.append_mode else '禁用'}")
        print(f"{'='*60}")

        processed = []
        for idx, md_file in enumerate(md_files, 1):
            print(f"\n[{idx}/{len(md_files)}]")
            try:
                out = self.process_markdown_file(str(md_file))
                processed.append(out)
            except Exception as e:
                print(f"  [ERROR] {md_file.name} - {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*60}")
        print(f"批量预处理完成！")
        print(f"成功: {len(processed)}/{len(md_files)}")
        print(f"{'='*60}")

        return processed


# ============================================================
# 快捷入口
# ============================================================

def preprocess_all(
    input_dir: str,
    pattern: str = "*_raw.md",
    recursive: bool = True,
    append_mode: bool = False,
) -> list[str]:
    """
    快捷入口：对目录下所有 raw MD 执行已注册的修复规则。

    Args:
        input_dir: MD文件所在目录
        pattern: 文件名匹配模式
        recursive: 是否递归
        append_mode: 追加模式，为 True 时跳过输出文件已存在的文件

    Returns:
        已处理的文件路径列表
    """
    processor = MarkdownPreprocessor(append_mode=append_mode)

    # ---- 注册修复规则（按执行顺序） ----
    processor.register_fix("修复主标题格式", fix_main_title)
    processor.register_fix("补全缺失的 1.2 目录标题", fix_missing_toc_header)
    processor.register_fix("图片引用上下加空行", fix_image_blank_lines)

    return processor.process_directory(input_dir, pattern=pattern, recursive=recursive)


# ============================================================
# CLI 入口（调试用）
# ============================================================

if __name__ == "__main__":
    MARKDOWN_DIR = str(Path(__file__).parent.parent / "markdown_mineru")

    append_mode = "--append" in sys.argv

    preprocess_all(MARKDOWN_DIR, recursive=True, append_mode=append_mode)
