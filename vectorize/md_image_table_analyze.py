"""
MD图片分析处理脚本 - 读取MD文件，分析其中的图片并更新MD

缓存机制：
- 首次分析图片时，会将大模型返回结果缓存到 images/{hash}_analysis.md
- 后续再执行时，如果缓存文件存在则直接读取，不再调用大模型
- 使用 --no-cache 参数可强制重新调用大模型
"""
import asyncio
import re
import sys
from pathlib import Path
from vision_service import get_vision_service


class MarkdownImageAnalysis:
    """Markdown图片分析处理器"""

    def __init__(self, model: str, use_cache: bool = True, append_mode: bool = False):
        """
        初始化分析处理器

        Args:
            model: VL模型名称
            use_cache: 是否使用缓存（默认True）
            append_mode: 追加模式，为 True 时跳过输出文件已存在的文件，只处理新文件
        """
        self.vision_service = get_vision_service(model=model)
        self.use_cache = use_cache
        self.append_mode = append_mode
        # 匹配 MinerU 输出的图片格式: ![](images/xxx.jpg)
        self.image_pattern = re.compile(
            r'!\[([^\]]*)\]\((images/[^)]+\.(?:jpg|png|webp))\)'
        )
        # 匹配 HTML 表格: <table>...</table>
        self.table_pattern = re.compile(
            r'<table>.*?</table>',
            re.DOTALL
        )
        self._image_id_counter = 0
        self._table_id_counter = 0

    def _get_cache_path(self, image_path: str, base_dir: Path) -> Path:
        """
        根据图片路径获取对应的缓存文件路径

        例如: images/abc123.jpg -> images/abc123_analysis.md

        Args:
            image_path: 图片相对路径
            base_dir: MD文件所在目录

        Returns:
            缓存文件完整路径
        """
        full_image_path = base_dir / image_path
        # 图片文件名（含扩展名），如 abc123.jpg
        image_filename = full_image_path.name
        # 缓存文件名: abc123_analysis.md
        cache_filename = f"{full_image_path.stem}_analysis.md"
        return full_image_path.parent / cache_filename

    def _read_cache(self, cache_path: Path) -> str | None:
        """
        读取缓存的分析结果

        Args:
            cache_path: 缓存文件路径

        Returns:
            缓存的分析文本，如果文件不存在或读取失败则返回 None
        """
        if not cache_path.exists():
            return None
        try:
            content = cache_path.read_text(encoding='utf-8').strip()
            if content:
                return content
        except Exception:
            pass
        return None

    def _write_cache(self, cache_path: Path, analysis_text: str):
        """
        将分析结果写入缓存文件

        Args:
            cache_path: 缓存文件路径
            analysis_text: 分析文本
        """
        try:
            cache_path.write_text(analysis_text, encoding='utf-8')
        except Exception as e:
            print(f"      ⚠️ 缓存写入失败: {e}")

    async def analyze_single_image(
        self,
        image_path: str,
        base_dir: Path,
        context: str = "这是基金年报中的图片"
    ) -> str:
        """
        分析单张图片（优先使用缓存）

        Args:
            image_path: 图片相对路径（如 images/page_1_img_0.png）
            base_dir: MD文件所在目录
            context: 上下文提示

        Returns:
            分析结果
        """
        # 构建完整路径
        full_path = base_dir / image_path

        if not full_path.exists():
            return f"[图片文件不存在: {image_path}]"

        # 检查缓存
        if self.use_cache:
            cache_path = self._get_cache_path(image_path, base_dir)
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        # 调用视觉服务分析
        result = await self.vision_service.analyze_image_from_file(
            str(full_path),
            context=context
        )

        # 写入缓存（仅当分析成功时）
        if self.use_cache and not result.startswith("[图片分析失败"):
            cache_path = self._get_cache_path(image_path, base_dir)
            self._write_cache(cache_path, result)

        return result

    def build_analysis_block(
        self,
        image_id: str,
        image_path: str,
        analysis_text: str
    ) -> str:
        """
        构建包含分析结果的图片块（用文字描述替代图片地址）
        """
        return (
            f"<!-- IMAGE_START id={image_id} -->\n\n"
            f"![图片]({image_path})\n\n"
            f"**图片内容分析:**\n\n{analysis_text}\n\n"
            f"<!-- IMAGE_END id={image_id} -->"
        )

    def build_table_block(
        self,
        table_id: str,
        table_html: str
    ) -> str:
        """
        构建包含标识的表格块
        """
        return (
            f"<!-- TABLE_START id={table_id} -->\n\n"
            f"{table_html}\n\n"
            f"<!-- TABLE_END id={table_id} -->"
        )

    async def process_markdown_file(
        self,
        md_path: str,
        output_path: str = None,
        context: str = "这是基金年报中的图片"
    ) -> str:
        """
        处理单个MD文件，分析其中的所有图片和表格

        Args:
            md_path: MD文件路径
            output_path: 输出路径（不指定则在同目录下创建新文件）
            context: 上下文提示

        Returns:
            输出文件路径
        """
        md_path = Path(md_path)
        base_dir = md_path.parent

        if output_path is None:
            # 去掉 _preprocessed 后缀，生成最终文件名
            stem = md_path.stem
            if stem.endswith("_preprocessed"):
                stem = stem[:-13]  # 移除 "_preprocessed"(13字符)
            output_path = base_dir / f"{stem}_analyzed.md"
        else:
            output_path = Path(output_path)

        print(f"\n{'='*60}")
        print(f"处理文件: {md_path.name}")
        print(f"{'='*60}")

        # 追加模式：输出文件已存在则跳过
        if self.append_mode and output_path.exists():
            print(f"  [SKIP] 输出文件已存在，追加模式跳过: {output_path.name}")
            return str(output_path)

        # 读取MD文件
        with open(md_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # 解析图片块（保存原始内容中的位置信息）
        image_matches = []
        self._image_id_counter = 0
        for match in self.image_pattern.finditer(original_content):
            self._image_id_counter += 1
            image_path = match.group(2)
            image_id = f"image_{self._image_id_counter}"
            image_matches.append({
                'type': 'image',
                'id': image_id,
                'path': image_path,
                'start': match.start(),
                'end': match.end(),
                'original': match.group(0)
            })

        # 解析表格块（保存原始内容中的位置信息）
        table_matches = []
        self._table_id_counter = 0
        for match in self.table_pattern.finditer(original_content):
            self._table_id_counter += 1
            table_id = f"table_{self._table_id_counter}"
            table_matches.append({
                'type': 'table',
                'id': table_id,
                'start': match.start(),
                'end': match.end(),
                'html': match.group(0)
            })

        if not image_matches and not table_matches:
            print("  未找到图片块和表格")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

            print(f"\n  ✓ 已保存: {output_path}")
            return str(output_path)

        print(f"  找到 {len(image_matches)} 张图片, {len(table_matches)} 个表格")

        # 分析所有图片，生成替换内容
        image_replacements = {}
        cache_hits = 0
        cache_misses = 0
        image_success_count = 0  # 成功分析的图片数量
        image_fail_count = 0     # 分析失败的图片数量
        for idx, img_match in enumerate(image_matches, 1):
            image_id = img_match['id']
            image_path = img_match['path']

            # 检查是否命中缓存
            from_cache = False
            if self.use_cache:
                cache_path = self._get_cache_path(image_path, base_dir)
                from_cache = self._read_cache(cache_path) is not None

            if from_cache:
                print(f"  [图片 {idx}/{len(image_matches)}] 缓存命中: {image_path}")
                cache_hits += 1
            else:
                print(f"  [图片 {idx}/{len(image_matches)}] 调用模型: {image_path}")
                cache_misses += 1

            # 分析图片
            analysis_text = await self.analyze_single_image(image_path, base_dir, context)

            # 检查是否分析失败
            if analysis_text.startswith("[图片分析失败") or analysis_text.startswith("[图片文件不存在"):
                print(f"      ⚠️ 处理失败，保留原图片块")
                print(analysis_text)
                # 保留原始内容
                image_replacements[image_id] = img_match['original']
                image_fail_count += 1
            else:
                # 构建新的图片块
                new_block = self.build_analysis_block(image_id, image_path, analysis_text)
                image_replacements[image_id] = new_block
                image_success_count += 1
                print(f"      ✓ 完成 ({len(analysis_text)} 字符)")

        if cache_hits > 0 or cache_misses > 0:
            print(f"  图片统计: {cache_hits} 缓存命中, {cache_misses} 调用模型")

        # 如果有图片需要分析但全部失败，则不生成输出文件
        if len(image_matches) > 0 and image_success_count == 0:
            print(f"\n  ⚠️ 所有图片分析均失败（{image_fail_count} 张），不生成 _analyzed 文件")
            return None

        # 合并所有需要替换的元素（图片+表格）
        all_replacements = []

        # 添加图片替换
        for img_match in image_matches:
            all_replacements.append({
                'start': img_match['start'],
                'end': img_match['end'],
                'new_content': image_replacements[img_match['id']]
            })

        # 添加表格替换
        for idx, table_match in enumerate(table_matches, 1):
            print(f"  [表格 {idx}/{len(table_matches)}] 标识表格 (id={table_match['id']})")
            new_block = self.build_table_block(table_match['id'], table_match['html'])
            all_replacements.append({
                'start': table_match['start'],
                'end': table_match['end'],
                'new_content': new_block
            })

        # 按位置从后往前排序，避免位置偏移
        all_replacements.sort(key=lambda x: x['start'], reverse=True)

        # 执行所有替换
        result_content = original_content
        for replacement in all_replacements:
            result_content = (
                result_content[:replacement['start']] +
                replacement['new_content'] +
                result_content[replacement['end']:]
            )

        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_content)

        print(f"\n  ✓ 已保存: {output_path}")
        return str(output_path)


async def process_markdown_directory(
    input_dir: str,
    context: str = "这是基金年报中的图片",
    pattern: str = "*_preprocessed.md",
    recursive: bool = True,
    use_cache: bool = True,
    append_mode: bool = False,
):
    """
    批量处理目录下的所有MD文件

    Args:
        input_dir: MD文件所在目录
        context: 上下文提示
        recursive: 是否递归处理子目录
        use_cache: 是否使用缓存
        append_mode: 追加模式，为 True 时跳过输出文件已存在的文件
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        print(f"错误: 目录不存在 - {input_dir}")
        return

    # 只查找待分析的原始 MD 文件
    if recursive:
        md_files = list(input_dir.rglob(pattern))
    else:
        md_files = list(input_dir.glob(pattern))

    if not md_files:
        print(f"在 {input_dir} 中未找到MD文件")
        return

    print(f"\n{'='*60}")
    print(f"批量分析处理")
    print(f"{'='*60}")
    print(f"目录: {input_dir}")
    print(f"找到 {len(md_files)} 个MD文件")
    print(f"缓存: {'启用' if use_cache else '禁用'}")
    print(f"追加模式: {'启用' if append_mode else '禁用'}")
    print(f"{'='*60}")

    processor = MarkdownImageAnalysis(model="qwen3.5-plus-2026-04-20", use_cache=use_cache, append_mode=append_mode)
    success_count = 0

    for idx, md_file in enumerate(md_files, 1):
        print(f"\n[{idx}/{len(md_files)}]")
        try:
            result = await processor.process_markdown_file(str(md_file), context=context)
            if result is not None:
                success_count += 1
        except Exception as e:
            print(f"  ✗ 错误: {md_file.name} - {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"批量处理完成！")
    print(f"成功: {success_count}/{len(md_files)}")
    print(f"{'='*60}")


if __name__ == "__main__":

    # 配置
    MARKDOWN_DIR = str(Path(__file__).parent.parent / "markdown_mineru")  # MD文件目录
    CONTEXT = "这是基金年报中的图片"  # 上下文提示
    USE_CACHE = "--no-cache" not in sys.argv  # 默认启用缓存，传 --no-cache 禁用
    APPEND_MODE = "--append" in sys.argv  # 追加模式，传 --append 启用

    # 运行批量处理
    asyncio.run(process_markdown_directory(
        MARKDOWN_DIR, CONTEXT,
        use_cache=USE_CACHE,
        append_mode=APPEND_MODE,
    ))

