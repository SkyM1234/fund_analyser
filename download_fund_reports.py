r"""下载 2025 年基金年度报告，并根据 PDF 正文校验基金信息。

常用命令（在项目根目录执行）：

    # 默认下载或断点续传，直到目录中包含 200 份报告
    E:/python_envs/pdf/python.exe download_fund_reports.py

    # 下载 10 份报告到单独目录，用于快速测试
    E:/python_envs/pdf/python.exe download_fund_reports.py --target-count 10 --output-dir annual_reports_test

    # 校验基金代码时扫描 PDF 的前 12 页
    E:/python_envs/pdf/python.exe download_fund_reports.py --scan-pages 12

    # 删除目标目录中已有的 PDF 和下载元数据，然后重新下载
    E:/python_envs/pdf/python.exe download_fund_reports.py --clean-output --target-count 200

    # 查看全部命令行参数
    E:/python_envs/pdf/python.exe download_fund_reports.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pymupdf
import requests


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "annual_reports_2025_funds"
TARGET_COUNT = 200
DEFAULT_SCAN_PAGES = 8
RECORD_VERSION = 3

ANNOUNCEMENT_API = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
FUND_LIST_API = "https://www.cninfo.com.cn/new/data/fund_stock.json"
PDF_BASE = "https://static.cninfo.com.cn/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.cninfo.com.cn/new/index",
    "Origin": "https://www.cninfo.com.cn",
    "X-Requested-With": "XMLHttpRequest",
}

SEARCH_KEYWORDS = [
    "科技 2025年年度报告",
    "信息技术 2025年年度报告",
    "人工智能 2025年年度报告",
    "半导体 2025年年度报告",
    "芯片 2025年年度报告",
    "新能源 2025年年度报告",
    "生物科技 2025年年度报告",
    "高端装备 2025年年度报告",
    "新材料 2025年年度报告",
    "云计算 2025年年度报告",
    "大数据 2025年年度报告",
    "物联网 2025年年度报告",
    "机器人 2025年年度报告",
    "通信 2025年年度报告",
    "电子 2025年年度报告",
    "软件 2025年年度报告",
    "互联网 2025年年度报告",
    "智能制造 2025年年度报告",
    "光电 2025年年度报告",
    "创新 2025年年度报告",
    "科创 2025年年度报告",
    "创业板 2025年年度报告",
    "数字经济 2025年年度报告",
    "先进制造 2025年年度报告",
    "智能汽车 2025年年度报告",
    "新能源车 2025年年度报告",
    "碳中和 2025年年度报告",
    "硬科技 2025年年度报告",
]

FUND_CODE_PATTERN = re.compile(r"基金(?:主)?代码\s*[：:]?\s*([0-9]{6})")
FUND_NAME_PATTERN = re.compile(r"基金名称\s+(.{2,120}?)\s+基金简称")
SHORT_NAME_PATTERN = re.compile(
    r"基金简称\s+(.{1,60}?)\s+"
    r"(?:场内简称|基金(?:主)?代码|基金运作方式|基金合同生效日)"
)


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


def sanitize_filename(name: str) -> str:
    name = strip_html(name)
    name = re.sub(r'[\\/:*?"<>|\r\n]', "_", name)
    return name.strip()[:240]


def is_2025_fund_annual_report(title: str) -> bool:
    title = strip_html(title)
    if "2025" not in title or "年度报告" not in title:
        return False
    excluded = (
        "摘要",
        "问询函",
        "回复",
        "更正",
        "补充",
        "英文版",
        "英文",
        "提示性",
        "修订版",
        "已取消",
        "取消",
        "说明",
    )
    return not any(word in title for word in excluded)


def search_fund_annual_reports(
    session: requests.Session, page_num: int, searchkey: str
) -> dict[str, Any]:
    data = {
        "pageNum": page_num,
        "pageSize": 30,
        "column": "fund",
        "tabName": "fulltext",
        "plate": "fund",
        "stock": "",
        "searchkey": searchkey,
        "secid": "",
        "category": "category_ndbg_fund",
        "trade": "",
        "seDate": "2026-01-01~2026-05-31",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    response = session.post(ANNOUNCEMENT_API, data=data, timeout=30)
    response.raise_for_status()
    return response.json()


def load_fund_names(session: requests.Session) -> dict[str, str]:
    try:
        response = session.get(FUND_LIST_API, timeout=30)
        response.raise_for_status()
        items = response.json().get("stockList") or []
    except (requests.RequestException, ValueError, AttributeError) as exc:
        print(f"  [WARN] Fund short-name list unavailable: {exc}")
        return {}
    return {
        str(item["code"]): str(item.get("zwjc") or "")
        for item in items
        if item.get("code")
    }


def api_fund_code(item: dict[str, Any]) -> str:
    match = re.search(r"\d{6}", str(item.get("secCode", "")))
    return match.group(0) if match else ""


def source_key(item: dict[str, Any]) -> str:
    return str(item.get("announcementId") or item.get("adjunctUrl") or "")


def download_pdf(
    session: requests.Session, adjunct_url: str, attempts: int = 3
) -> bytes | None:
    pdf_url = urljoin(PDF_BASE, adjunct_url)
    for attempt in range(attempts):
        try:
            response = session.get(pdf_url, timeout=90)
            content = response.content
            if (
                response.status_code == 200
                and len(content) > 10_000
                and content.startswith(b"%PDF")
            ):
                return content
            print(
                f"  [RETRY {attempt + 1}] status={response.status_code} "
                f"size={len(content)}"
            )
        except requests.RequestException as exc:
            print(f"  [RETRY {attempt + 1}] {exc}")
        if attempt + 1 < attempts:
            time.sleep(2)
    return None


def ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def extract_pdf_identity(content: bytes, scan_pages: int) -> dict[str, Any]:
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            pages_scanned = min(scan_pages, len(document))
            text = "\n".join(
                document[index].get_text("text", sort=True)
                for index in range(pages_scanned)
            )
            page_count = len(document)
    except Exception as exc:
        return {
            "body_primary_code": "",
            "body_codes": [],
            "body_fund_name": "",
            "body_short_name": "",
            "page_count": 0,
            "pages_scanned": 0,
            "extracted_chars": 0,
            "extraction_error": f"{type(exc).__name__}: {exc}",
        }

    normalized = re.sub(r"\s+", " ", text).strip()
    body_codes = ordered_unique(FUND_CODE_PATTERN.findall(text))
    fund_name_match = FUND_NAME_PATTERN.search(normalized)
    short_name_match = SHORT_NAME_PATTERN.search(normalized)
    return {
        "body_primary_code": body_codes[0] if body_codes else "",
        "body_codes": body_codes,
        "body_fund_name": fund_name_match.group(1).strip() if fund_name_match else "",
        "body_short_name": (
            short_name_match.group(1).strip() if short_name_match else ""
        ),
        "page_count": page_count,
        "pages_scanned": pages_scanned,
        "extracted_chars": len(text.strip()),
        "extraction_error": "",
    }


def validation_status(api_code: str, identity: dict[str, Any]) -> str:
    if identity["extraction_error"]:
        return "pdf_read_error"
    body_code = identity["body_primary_code"]
    if not body_code:
        return "body_code_missing"
    if not api_code:
        return "api_code_missing"
    if body_code != api_code:
        return "code_mismatch"
    return "match"


def empty_record() -> dict[str, Any]:
    return {
        "version": RECORD_VERSION,
        "downloaded": 0,
        "seen_sources": [],
        "content_hashes": {},
        "files": [],
    }


def load_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_record()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_record()
    if record.get("version") != RECORD_VERSION:
        print("  [WARN] Ignoring legacy record with a different validation format.")
        print("  [WARN] Use --clean-output when rebuilding an existing output directory.")
        return empty_record()
    hashes = record.get("content_hashes")
    if not isinstance(hashes, dict):
        record["content_hashes"] = {}
    else:
        record["content_hashes"] = {
            digest: files if isinstance(files, list) else [files]
            for digest, files in hashes.items()
        }
    return record


def build_review(record: dict[str, Any]) -> dict[str, Any]:
    files = record["files"]
    mismatches = [item for item in files if item["validation"] == "code_mismatch"]
    extraction_issues = [
        item
        for item in files
        if item["validation"] in {"pdf_read_error", "body_code_missing"}
    ]
    duplicate_groups = [
        {"sha256": digest, "files": filenames}
        for digest, filenames in record["content_hashes"].items()
        if len(filenames) > 1
    ]
    return {
        "summary": {
            "downloaded": len(files),
            "matches": sum(item["validation"] == "match" for item in files),
            "code_mismatches": len(mismatches),
            "extraction_issues": len(extraction_issues),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_files": sum(len(group["files"]) for group in duplicate_groups),
        },
        "code_mismatches": mismatches,
        "extraction_issues": extraction_issues,
        "duplicate_groups": duplicate_groups,
    }


def save_state(
    record_path: Path,
    review_path: Path,
    record: dict[str, Any],
    seen_sources: set[str],
) -> None:
    record["downloaded"] = len(record["files"])
    record["seen_sources"] = sorted(seen_sources)
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    review_path.write_text(
        json.dumps(build_review(record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def available_filepath(
    output_dir: Path, filename: str, source: str
) -> tuple[Path, str]:
    path = output_dir / filename
    if not path.exists():
        return path, filename
    stem = Path(filename).stem
    suffix = re.sub(r"\W+", "", source)[-12:] or "duplicate"
    candidate = f"{stem}_{suffix}.pdf"
    return output_dir / candidate, candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  默认下载或断点续传至 200 份：
    E:/python_envs/pdf/python.exe download_fund_reports.py

  下载 10 份报告到单独的测试目录：
    E:/python_envs/pdf/python.exe download_fund_reports.py --target-count 10 --output-dir annual_reports_test

  使用 PDF 的前 12 页校验基金代码：
    E:/python_envs/pdf/python.exe download_fund_reports.py --scan-pages 12

  删除输出目录中的已有 PDF 和元数据后重新下载：
    E:/python_envs/pdf/python.exe download_fund_reports.py --clean-output --target-count 200
""",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument("--scan-pages", type=int, default=DEFAULT_SCAN_PAGES)
    parser.add_argument("--download-delay", type=float, default=0.3)
    parser.add_argument("--reset-record", action="store_true")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete PDFs and downloader metadata in the selected output directory.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.target_count <= 0:
        parser.error("--target-count must be greater than zero")
    if args.scan_pages <= 0:
        parser.error("--scan-pages must be greater than zero")
    if args.download_delay < 0:
        parser.error("--download-delay cannot be negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.output_dir / "_download_record.json"
    review_path = args.output_dir / "_pdf_review.json"
    if args.clean_output:
        for pdf_path in args.output_dir.glob("*.pdf"):
            pdf_path.unlink()
        record_path.unlink(missing_ok=True)
        review_path.unlink(missing_ok=True)

    record = (
        empty_record()
        if args.reset_record or args.clean_output
        else load_record(record_path)
    )
    seen_sources = set(record["seen_sources"])
    attempted_sources: set[str] = set()

    session = requests.Session()
    session.headers.update(HEADERS)
    fund_names = load_fund_names(session)
    print(f"Loaded {len(fund_names)} fund short names")

    for keyword in SEARCH_KEYWORDS:
        if len(record["files"]) >= args.target_count:
            break
        print(f'\n--- Keyword: "{keyword}" ---')
        page = 1

        while len(record["files"]) < args.target_count:
            try:
                result = search_fund_annual_reports(session, page, keyword)
            except (requests.RequestException, ValueError) as exc:
                print(f"  [ERR] Search failed: {exc}")
                time.sleep(3)
                continue

            announcements = result.get("announcements") or []
            total_pages = int(result.get("totalpages") or 0)
            if not announcements:
                break
            print(f"  Page {page}/{total_pages}, found {len(announcements)} items")

            for item in announcements:
                if len(record["files"]) >= args.target_count:
                    break
                title = strip_html(item.get("announcementTitle", ""))
                adjunct_url = item.get("adjunctUrl", "")
                current_source = source_key(item)
                if (
                    not adjunct_url
                    or not current_source
                    or current_source in seen_sources
                    or current_source in attempted_sources
                    or not is_2025_fund_annual_report(title)
                ):
                    continue
                attempted_sources.add(current_source)

                code = api_fund_code(item)
                short_name = (
                    fund_names.get(code)
                    or strip_html(str(item.get("secName", ""))).split(",")[0]
                    or "unknown"
                )
                if args.dry_run:
                    print(f"  [DRY RUN] {code or 'unknown'} {short_name}: {title[:100]}")
                    continue

                print(
                    f"  [{len(record['files']) + 1}/{args.target_count}] "
                    f"{code or 'unknown'} {short_name}"
                )
                content = download_pdf(session, adjunct_url)
                if content is None:
                    print("  [FAIL] Download failed; it will be retried on the next run")
                    continue

                digest = hashlib.sha256(content).hexdigest()
                duplicate_of = list(record["content_hashes"].get(digest, []))
                identity = extract_pdf_identity(content, args.scan_pages)
                status = validation_status(code, identity)

                filename = sanitize_filename(f"{code}_{short_name}_{title}") + ".pdf"
                filepath, filename = available_filepath(
                    args.output_dir, filename, current_source
                )
                filepath.write_bytes(content)

                record["content_hashes"].setdefault(digest, []).append(filename)
                record["files"].append(
                    {
                        "filename": filename,
                        "title": title,
                        "announcement_id": item.get("announcementId", ""),
                        "adjunct_url": adjunct_url,
                        "api_code": code,
                        "api_short_name": short_name,
                        "org_id": item.get("orgId", ""),
                        "sha256": digest,
                        "duplicate": bool(duplicate_of),
                        "duplicate_of": duplicate_of,
                        "validation": status,
                        **identity,
                    }
                )
                seen_sources.add(current_source)
                save_state(record_path, review_path, record, seen_sources)

                if duplicate_of:
                    print(f"  [DUPLICATE] same content as {', '.join(duplicate_of)}")
                if status != "match":
                    print(
                        f"  [REVIEW] {status}: API={code or '-'} "
                        f"BODY={identity['body_primary_code'] or '-'}"
                    )
                time.sleep(args.download_delay)

            if total_pages and page >= total_pages:
                break
            page += 1
            time.sleep(0.5)

    save_state(record_path, review_path, record, seen_sources)
    review = build_review(record)
    summary = review["summary"]
    print(f"\nDone: {summary['downloaded']} reports in {args.output_dir}")
    print(
        "Review: "
        f"{summary['code_mismatches']} code mismatches, "
        f"{summary['extraction_issues']} extraction issues, "
        f"{summary['duplicate_groups']} duplicate groups"
    )


if __name__ == "__main__":
    main()
