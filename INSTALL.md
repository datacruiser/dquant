# DQuant 安装指南

## 快速开始

### 1. 克隆仓库

```bash
cd ~/github
git clone https://github.com/yourusername/dquant.git
cd dquant
```

### 2. 安装依赖

**方式 A: 使用安装脚本 (推荐)**

```bash
./install.sh
```

**方式 B: 手动安装**

```bash
# 最小安装
pip install -r requirements-minimal.txt

# 标准安装 (推荐)
pip install -r requirements.txt

# 完整安装
pip install -r requirements-full.txt
```

**方式 C: 使用 pip**

```bash
# 基础安装
pip install -e .

# 安装可选依赖
pip install -e ".[ai]"      # 机器学习
pip install -e ".[data]"    # 数据源
pip install -e ".[dev]"     # 开发工具
```

### 3. 验证安装

```bash
python -c "from dquant import Engine; print('✓ 安装成功')"
```

## 依赖详情

### 核心依赖 (必需)

| 包名 | 版本 | 用途 |
|------|------|------|
| numpy | >=1.19.0 | 数值计算 |
| pandas | >=1.1.0 | 数据处理 |

### 可视化 (推荐)

| 包名 | 版本 | 用途 |
|------|------|------|
| matplotlib | >=3.3.0 | 图表绘制 |
| seaborn | >=0.11.0 | 热力图等 |

### 数据源 (按需)

| 包名 | 版本 | 数据源 | 费用 |
|------|------|--------|------|
| akshare | >=1.10.0 | A股 | 免费 |
| tushare | >=1.2.0 | A股 | 需要 token |
| yfinance | >=0.1.70 | 美股/全球 | 免费 |
| pytdx | >=1.72 | 通达信 | 免费 |
| jqdatasdk | >=1.8.0 | 聚宽 | 需要账号 |
| rqdatac | >=2.9.0 | 米筐 | 需要账号 |

### 机器学习 (按需)

| 包名 | 版本 | 用途 |
|------|------|------|
| xgboost | >=1.3.0 | 梯度提升 |
| lightgbm | >=3.0.0 | 梯度提升 |
| scikit-learn | >=0.24.0 | 因子处理 |

### 强化学习 (按需)

| 包名 | 版本 | 用途 |
|------|------|------|
| gym | >=0.18.0 | 环境接口 |
| torch | >=1.8.0 | 深度学习 |

### 数据库 (按需)

| 包名 | 版本 | 用途 |
|------|------|------|
| sqlalchemy | >=1.3.0 | SQL 数据库 |
| pymongo | >=3.11.0 | MongoDB |

## 安装场景

### 场景 1: 仅回测

```bash
pip install -r requirements-minimal.txt
pip install matplotlib
```

### 场景 2: A股研究

```bash
pip install -r requirements.txt
pip install akshare tushare
```

### 场景 3: 美股研究

```bash
pip install -r requirements.txt
pip install yfinance
```

### 场景 4: ML 因子挖掘

```bash
pip install -r requirements.txt
pip install xgboost lightgbm scikit-learn
```

### 场景 5: 强化学习

```bash
pip install -r requirements-full.txt
```

### 场景 6: 完整开发

```bash
pip install -r requirements-full.txt
pip install -e ".[dev]"
```

## 常见问题

### Q: 安装 akshare 失败？

```bash
# 尝试使用清华源
pip install akshare -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 安装 lightgbm 失败？

```bash
# macOS
brew install cmake libomp
pip install lightgbm

# Linux
pip install lightgbm --install-option=--nomp
```

### Q: 安装 torch 太大？

```bash
# 仅安装 CPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Q: 权限问题？

```bash
# 使用用户安装
pip install --user -r requirements.txt
```

## Python 版本

- **最低**: Python 3.8
- **推荐**: Python 3.9 或 3.10
- **不支持**: Python 3.7 及以下

检查 Python 版本:

```bash
python --version
```

## 虚拟环境 (推荐)

### venv

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### conda

```bash
conda create -n dquant python=3.9
conda activate dquant

pip install -r requirements.txt
```

## 更新

```bash
cd ~/github/dquant
git pull
pip install -e . --upgrade
```

## 下一步

安装完成后，可以:

1. 运行示例: `python examples/simple_backtest.py`
2. 查看 README: `cat README.md`
3. 开始使用: `from dquant import Engine`
