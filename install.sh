#!/bin/bash
# DQuant 快速安装脚本

set -e

echo "================================================"
echo "DQuant 安装脚本"
echo "================================================"
echo ""

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python 版本: $PYTHON_VERSION"

# 检查 pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 未安装"
    exit 1
fi
echo "✓ pip3 已安装"
echo ""

# 选择安装模式
echo "选择安装模式:"
echo "  1) 最小安装 (仅核心)"
echo "  2) 标准安装 (推荐)"
echo "  3) 完整安装 (所有功能)"
echo "  4) 开发安装 (包含测试工具)"
echo ""
read -p "请选择 [1-4, 默认=2]: " choice
choice=${choice:-2}

case $choice in
    1)
        echo "安装最小依赖..."
        pip3 install -r requirements-minimal.txt
        ;;
    2)
        echo "安装标准依赖..."
        pip3 install -r requirements.txt
        ;;
    3)
        echo "安装完整依赖..."
        pip3 install -r requirements-full.txt
        ;;
    4)
        echo "安装开发依赖..."
        pip3 install -r requirements-full.txt
        pip3 install -e ".[dev]"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "安装 DQuant 包..."
echo "================================================"
pip3 install -e .

echo ""
echo "================================================"
echo "✓ 安装完成!"
echo "================================================"
echo ""
echo "测试安装:"
echo "  python -c 'from dquant import Engine; print(\"OK\")'"
echo ""
echo "运行示例:"
echo "  cd examples"
echo "  python simple_backtest.py"
echo ""
