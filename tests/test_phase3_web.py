"""
Phase 3 Step 6: Web 模块安全导入测试
"""

import pytest


class TestWebModuleImport:

    def test_import_web_does_not_crash(self):
        """导入 web 模块不会崩溃"""
        import dquant.web

        # app 和 run_server 可能为 None（未实现）
        assert hasattr(dquant.web, "app")
        assert hasattr(dquant.web, "run_server")

    def test_web_app_is_none_when_not_implemented(self):
        """app 为 None 表示 Web 模块未实现"""
        import dquant.web

        # 当前 web/app.py 不存在，所以 app 应为 None
        assert dquant.web.app is None
        assert dquant.web.run_server is None

    def test_web_all_list(self):
        """__all__ 包含正确的导出名"""
        import dquant.web

        assert "app" in dquant.web.__all__
        assert "run_server" in dquant.web.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
