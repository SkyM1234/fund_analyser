"""
批量调用 MinerU API 解析 PDF，输出 Markdown + 图片。
每个 PDF 独立文件夹，内含 .md 和 images/（仅保留正文引用的图片，表格截图自动剔除）。
"""

import io
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import httpx

# === 配置 ===
INPUT_DIR = Path(__file__).parent.parent / "annual_reports_2025_funds"
OUTPUT_DIR = Path(__file__).parent.parent / "markdown_mineru"
API_URL = "http://localhost:8000/file_parse"
END_PAGES = 1000
TIMEOUT = 1800  # 单个 PDF 最长 30 分钟
CONTAINER_NAME = "mineru-api"  # MinerU API 容器名
RESTART_EVERY_N = 1  # 每处理 N 个 PDF 后重启容器（1 = 每个都重启，释放内存）
HEALTH_CHECK_URL = "http://localhost:8000/health"

_shutdown_event = threading.Event()  # 使用线程事件代替全局标志


def restart_mineru_container():
    """重启 MinerU Docker 容器，等待健康检查通过。"""
    print(f"[容器] 重启 {CONTAINER_NAME} ...")
    subprocess.run(
        ["docker", "restart", CONTAINER_NAME],
        capture_output=True,
        timeout=30,
    )
    # 等待健康检查通过
    print(f"[容器] 等待健康检查 {HEALTH_CHECK_URL} ...")
    for _ in range(120):  # 最多等 2 分钟
        if _shutdown_event.is_set():
            return False
        try:
            r = httpx.get(HEALTH_CHECK_URL, timeout=5)
            if r.status_code == 200:
                print(f"[容器] 就绪")
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"[容器] 警告: 健康检查超时，继续尝试", file=sys.stderr)
    return True  # 即使超时也继续，让请求本身去报错


def parse_pdf(pdf_path: Path) -> dict | None:
    """调用 MinerU API 解析 PDF，返回 {name, md, zip_bytes}。"""
    pdf_name = pdf_path.name
    print(f"[开始] {pdf_name}")

    form = {
        "lang_list": ["ch"],
        "backend": "hybrid-engine",
        "effort": "medium",
        "parse_method": "auto",
        "formula_enable": "false",
        "table_enable": "true",
        "image_analysis": "false",
        "server_url": "",
        "return_md": "true",
        "return_middle_json": "false",
        "return_model_output": "false",
        "return_content_list": "false",
        "return_images": "true",
        "response_format_zip": "true",
        "return_original_file": "false",
        "client_side_output_generation": "false",
        "start_page_id": "0",
        "end_page_id": str(END_PAGES),
    }

    # 使用较短的超时和定期检查，以便能够响应中断
    result = {"response": None, "error": None}

    def _make_request():
        """在单独线程中执行 HTTP 请求"""
        try:
            with open(pdf_path, "rb") as f:
                files = {"files": (pdf_name, f, "application/pdf")}
                # 使用原始的长超时，因为 MinerU 需要很长时间
                with httpx.Client(timeout=TIMEOUT) as client:
                    result["response"] = client.post(API_URL, data=form, files=files)
        except Exception as e:
            result["error"] = e

    # 在单独线程中启动请求
    thread = threading.Thread(target=_make_request, daemon=True)
    thread.start()

    # 等待请求完成，同时定期检查中断信号
    start_time = time.time()
    while thread.is_alive():
        if _shutdown_event.is_set():
            print(f"[中断] 取消 {pdf_name}")
            return None

        # 检查是否超时
        if time.time() - start_time > TIMEOUT:
            print(f"[超时] {pdf_name}: 超过 {TIMEOUT} 秒")
            return None

        # 每 0.5 秒检查一次
        thread.join(timeout=0.5)

    # 处理结果
    if result["error"]:
        print(f"[失败] {pdf_name}: {result['error']}", file=sys.stderr)
        return None

    response = result["response"]
    if response is None:
        print(f"[失败] {pdf_name}: 无响应")
        return None

    if response.status_code != 200:
        print(f"[失败] {pdf_name}: HTTP {response.status_code}")
        return None

    zip_bytes = response.content
    print(f"[完成] {pdf_name}  → 响应 {len(zip_bytes)} bytes")
    return {"name": pdf_name, "zip_bytes": zip_bytes}


def save_result(result: dict, output_dir: Path):
    """解压 ZIP → 提取 markdown + 图片，保存到子文件夹。"""
    pdf_stem = Path(result["name"]).stem
    sub_dir = output_dir / pdf_stem
    sub_dir.mkdir(parents=True, exist_ok=True)

    zip_bytes = result["zip_bytes"]
    extracted_md = None

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        # 找出 markdown 文件（可能嵌套在子目录中）
        md_names = [n for n in zf.namelist() if n.endswith(".md")]
        if md_names:
            extracted_md = zf.read(md_names[0]).decode("utf-8")

        # 找出所有图片文件（路径中包含 /images/ 或 images/ ）
        all_images = [n for n in zf.namelist() if "/images/" in n or n.startswith("images/")]

        # 先把 Gradio API 图片路径替换为本地路径（必须在提取引用之前）
        if extracted_md:
            extracted_md = re.sub(
                r'!\[([^\]]*)\]\(/gradio_api/file=[^)]+?/images/([^/)]+)\)',
                r'![\1](images/\2)',
                extracted_md,
            )

        if extracted_md:
            # 找出 markdown 中实际引用的图片文件名
            referenced = set()
            referenced.update(re.findall(r'!\[.*?\]\(images/([^)]+)\)', extracted_md))
            referenced.update(re.findall(r'<img[^>]+src=["\']images/([^"\']+)["\']', extracted_md))
        else:
            referenced = set()

        # 只提取被引用的图片（直接读文件内容写出，避免嵌套目录问题）
        extracted_count = 0
        for member in all_images:
            filename = Path(member).name
            if filename in referenced:
                # 创建 images/ 子目录并直接写出文件
                images_dir = sub_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                images_dir.joinpath(filename).write_bytes(zf.read(member))
                extracted_count += 1

        # 保存 markdown
        md_path = sub_dir / f"{pdf_stem}_raw.md"
        md_path.write_text(extracted_md or "", encoding="utf-8")

    # 清理空 images 目录
    images_dir = sub_dir / "images"
    if images_dir.exists() and not list(images_dir.iterdir()):
        images_dir.rmdir()

    missing = len(referenced) - extracted_count
    if missing > 0:
        print(f"  -> 图片: {extracted_count}张, 缺失{missing}张")
    else:
        print(f"  -> 图片: {extracted_count}张")


def main():
    # 启动键盘监听线程
    def _keyboard_listener():
        """监听键盘输入，检测 Ctrl+C"""
        try:
            while not _shutdown_event.is_set():
                if input() == "":  # 任意输入都会触发
                    continue
        except (KeyboardInterrupt, EOFError):
            print("\n[中断] 检测到 Ctrl+C，正在停止...")
            _shutdown_event.set()

    listener_thread = threading.Thread(target=_keyboard_listener, daemon=True)
    listener_thread.start()

    # 追加模式：仅创建输出目录，不清空
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"错误: {INPUT_DIR} 中没有找到 PDF 文件")
        sys.exit(1)

    # 跳过已有输出
    skipped = []
    pending = []
    for pdf_path in pdf_files:
        if (OUTPUT_DIR / pdf_path.stem).exists():
            skipped.append(pdf_path.name)
        else:
            pending.append(pdf_path)

    if skipped:
        print(f"跳过 {len(skipped)} 个已完成")

    if not pending:
        print("所有 PDF 均已处理。")
        return

    print(f"待处理 {len(pending)} 个 PDF | 输出: {OUTPUT_DIR}")
    print(f"提示: 按 Ctrl+C 可随时中断")
    print("-" * 50)

    success = fail = 0
    start = time.time()

    try:
        for pdf_path in pending:
            if _shutdown_event.is_set():
                print("[中断] 停止处理剩余文件")
                break

            result = parse_pdf(pdf_path)

            if result:
                try:
                    save_result(result, OUTPUT_DIR)
                    success += 1
                except Exception as e:
                    print(f"[保存失败] {result['name']}: {e}", file=sys.stderr)
                    fail += 1
            else:
                fail += 1

            done = success + fail

            # 每处理 N 个 PDF 后重启容器以释放内存
            if done % RESTART_EVERY_N == 0:
                restart_mineru_container()
            if done == 0:
                break
            elapsed = time.time() - start
            avg = elapsed / done
            remaining = avg * (len(pending) - done)
            print(f"[进度] {done}/{len(pending)}  成功:{success}  失败:{fail}  "
                  f"已用:{elapsed/60:.1f}分  剩余约:{remaining/60:.1f}分")

    except KeyboardInterrupt:
        print("\n[中断] 检测到 Ctrl+C，正在停止...")
        _shutdown_event.set()

    total = time.time() - start
    print("-" * 50)
    print(f"完成! 成功:{success}  失败:{fail}  总耗时:{total/60:.1f}分")


if __name__ == "__main__":
    main()
