"""
下载BGE-M3和BGE-Reranker-v2-m3模型到本地embedding_model目录
"""
from pathlib import Path
from huggingface_hub import snapshot_download


def download_bge_m3_model(local_dir: str = "./embedding_model/bge-m3"):
    """
    下载BGE-M3模型到本地目录

    Args:
        local_dir: 本地存储路径
    """
    # 创建目录
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("开始下载BGE-M3模型")
    print("="*60)
    print(f"目标路径: {local_path.absolute()}")
    print(f"模型仓库: BAAI/bge-m3")
    print()

    # 检查是否已存在
    if (local_path / "config.json").exists():
        print("检测到模型已存在，是否重新下载？")
        choice = input("输入 'y' 重新下载，其他键跳过: ").strip().lower()
        if choice != 'y':
            print("跳过下载")
            return

    try:
        print("正在下载模型文件...")
        print("(模型约2GB，首次下载可能需要较长时间)")
        print()

        # 下载模型
        snapshot_download(
            repo_id="BAAI/bge-m3",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print()
        print("="*60)
        print("✓ BGE-M3模型下载成功!")
        print("="*60)
        print(f"模型位置: {local_path.absolute()}")

    except Exception as e:
        print()
        print("="*60)
        print("✗ 下载失败")
        print("="*60)
        print(f"错误信息: {e}")
        print()
        print("解决方案:")
        print("1. 检查网络连接")
        print("2. 使用镜像站点下载:")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        print("   然后重新运行此脚本")
        print("3. 手动下载模型:")
        print("   从 https://huggingface.co/BAAI/bge-m3 下载所有文件")
        print(f"   放到 {local_path.absolute()} 目录下")


def download_bge_reranker_model(local_dir: str = "./embedding_model/bge-reranker-v2-m3"):
    """
    下载BGE-Reranker-v2-m3模型到本地目录

    Args:
        local_dir: 本地存储路径
    """
    # 创建目录
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    print()
    print("="*60)
    print("开始下载BGE-Reranker-v2-m3模型")
    print("="*60)
    print(f"目标路径: {local_path.absolute()}")
    print(f"模型仓库: BAAI/bge-reranker-v2-m3")
    print()

    # 检查是否已存在
    if (local_path / "config.json").exists():
        print("检测到模型已存在，是否重新下载？")
        choice = input("输入 'y' 重新下载，其他键跳过: ").strip().lower()
        if choice != 'y':
            print("跳过下载")
            return

    try:
        print("正在下载模型文件...")
        print("(模型约2GB，首次下载可能需要较长时间)")
        print()

        # 下载模型
        snapshot_download(
            repo_id="BAAI/bge-reranker-v2-m3",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print()
        print("="*60)
        print("✓ BGE-Reranker-v2-m3模型下载成功!")
        print("="*60)
        print(f"模型位置: {local_path.absolute()}")

    except Exception as e:
        print()
        print("="*60)
        print("✗ 下载失败")
        print("="*60)
        print(f"错误信息: {e}")
        print()
        print("解决方案:")
        print("1. 检查网络连接")
        print("2. 使用镜像站点下载:")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        print("   然后重新运行此脚本")
        print("3. 手动下载模型:")
        print("   从 https://huggingface.co/BAAI/bge-reranker-v2-m3 下载所有文件")
        print(f"   放到 {local_path.absolute()} 目录下")


def check_model(local_dir: str = "./embedding_model/bge-m3"):
    """
    检查模型文件完整性

    Args:
        local_dir: 模型目录
    """
    local_path = Path(local_dir)

    print("="*60)
    print(f"检查模型文件: {local_path.name}")
    print("="*60)

    required_files = [
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
        "model.safetensors" if local_path.name == "bge-m3" else "pytorch_model.bin"
    ]

    if local_path.name == "bge-m3":
        required_files.append("1_Pooling/config.json")

    missing_files = []
    for file in required_files:
        file_path = local_path / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {file} ({size:.2f} MB)")
        else:
            print(f"✗ {file} (缺失)")
            missing_files.append(file)

    print()
    if not missing_files:
        print("✓ 模型文件完整，可以正常使用")
    else:
        print(f"✗ 缺失 {len(missing_files)} 个文件，请重新下载")
    print("="*60)


if __name__ == "__main__":
    import sys

    print("="*60)
    print("BGE模型下载工具")
    print("="*60)
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # 检查模式
        check_model(str(Path(__file__).parent.parent / "embedding_model" / "bge-m3"))
        check_model(str(Path(__file__).parent.parent / "embedding_model" / "bge-reranker-v2-m3"))
    else:
        # 下载模式
        print("将下载以下模型:")
        print("1. BGE-M3 (Embedding模型，约2GB)")
        print("2. BGE-Reranker-v2-m3 (Reranker模型，约2GB)")
        print()
        choice = input("是否继续? (y/n): ").strip().lower()
        if choice != 'y':
            print("取消下载")
            exit(0)

        # 下载两个模型
        download_bge_m3_model()
        download_bge_reranker_model()

        print()
        print("="*60)
        print("所有模型下载完成!")
        print("="*60)
        print("现在可以运行: python embedding_service.py")

