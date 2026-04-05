"""
DQuant Web 模块

Web 仪表盘和 API 服务（实验性功能）。

使用前需安装 FastAPI: pip install fastapi uvicorn
"""

try:
    from dquant.web.app import app, run_server

    __all__ = ["app", "run_server"]
except ImportError:
    # Web 模块未实现或缺少依赖
    app = None
    run_server = None
    __all__ = ["app", "run_server"]
