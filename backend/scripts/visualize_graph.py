"""
Graph可视化工具测试脚本

运行方式：
    # Docker 内运行
    docker exec fund-backend python -m scripts.visualize_graph
    # 裸机运行（在项目根目录下）
    python backend/scripts/visualize_graph.py
"""
import sys
import logging
from pathlib import Path

# 添加项目路径（scripts/ 的父目录即 backend 根目录）
backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.agent.multi_agent_controller import (
    print_graph_structure,
    export_graph_to_mermaid,
    export_graph_to_png,
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)

logger = logging.getLogger(__name__)

project_root = backend_root.parent  # 项目根目录


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🎨 Multi-Agent Graph Visualization Tool")
    print("="*70 + "\n")
    
    # 1. 打印结构
    print("📊 Step 1: Printing graph structure...")
    print_graph_structure()

    # 2. 导出Mermaid
    mermaid_file = project_root / "backend" / "docs" / "graph_mermaid.md"
    print(f"📄 Step 2: Exporting Mermaid diagram to {mermaid_file.relative_to(project_root)}...")
    mermaid_str = export_graph_to_mermaid(output_file=str(mermaid_file))
    
    if mermaid_str:
        print(f"   ✓ Success! ({len(mermaid_str)} characters)")
        print(f"   ✓ File saved: {mermaid_file}")
    else:
        print(f"   ✗ Failed to export Mermaid")
    
    # 3. 导出PNG
    png_file = project_root / "backend" / "docs" / "graph_architecture.png"
    print(f"\n🖼️  Step 3: Exporting PNG diagram to {png_file.relative_to(project_root)}...")
    
    try:
        success = export_graph_to_png(output_file=str(png_file))
        if success:
            print(f"   ✓ Success!")
            print(f"   ✓ File saved: {png_file}")
        else:
            print(f"   ⚠️  PNG export not available (missing dependencies)")
            print(f"   ℹ️  Install with: pip install pygraphviz")
    except Exception as e:
        print(f"   ⚠️  PNG export failed: {e}")
        print(f"   ℹ️  This is optional, Mermaid diagram is sufficient")
    
    # 4. 总结
    print("\n" + "="*70)
    print("✅ Visualization Complete!")
    print("="*70)
    print("\n📁 Generated files:")
    if mermaid_file.exists():
        print(f"  ✓ {mermaid_file.relative_to(project_root)}")
    if png_file.exists():
        print(f"  ✓ {png_file.relative_to(project_root)}")
    
    print("\n💡 Next steps:")
    print("  1. View the Mermaid diagram in your Markdown editor")
    print("  2. Commit the generated files to version control")
    print("  3. Include them in your documentation\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
