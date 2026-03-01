"""
测试交易安全模块
"""

import pytest
from datetime import datetime
from dquant.broker.base import Order
from dquant.broker.safety import (
    OrderValidator,
    FundChecker,
    TradingTimeChecker,
    TradingSafety,
)


class TestOrderValidator:
    """测试订单验证器"""
    
    def test_validate_symbol_valid(self):
        """测试有效的股票代码"""
        valid, msg = OrderValidator.validate_symbol('000001.SZ')
        assert valid is True
        
        valid, msg = OrderValidator.validate_symbol('600000.SH')
        assert valid is True
        
        valid, msg = OrderValidator.validate_symbol('300001.SZ')
        assert valid is True
    
    def test_validate_symbol_invalid(self):
        """测试无效的股票代码"""
        # 空代码
        valid, msg = OrderValidator.validate_symbol('')
        assert valid is False
        
        # 格式错误
        valid, msg = OrderValidator.validate_symbol('000001')
        assert valid is False
        
        # 不支持的市场
        valid, msg = OrderValidator.validate_symbol('000001.US')
        assert valid is False
    
    def test_validate_quantity_valid(self):
        """测试有效的数量"""
        valid, msg = OrderValidator.validate_quantity(100)
        assert valid is True
        
        valid, msg = OrderValidator.validate_quantity(1000)
        assert valid is True
    
    def test_validate_quantity_invalid(self):
        """测试无效的数量"""
        # 不是100的整数倍
        valid, msg = OrderValidator.validate_quantity(150)
        assert valid is False
        
        # 小于0
        valid, msg = OrderValidator.validate_quantity(-100)
        assert valid is False
        
        # 过大
        valid, msg = OrderValidator.validate_quantity(2000000)
        assert valid is False
    
    def test_validate_price_valid(self):
        """测试有效的价格"""
        valid, msg = OrderValidator.validate_price(10.5, 'LIMIT')
        assert valid is True
        
        # 市价单不需要价格
        valid, msg = OrderValidator.validate_price(None, 'MARKET')
        assert valid is True
    
    def test_validate_price_invalid(self):
        """测试无效的价格"""
        # 限价单没有价格
        valid, msg = OrderValidator.validate_price(None, 'LIMIT')
        assert valid is False
        
        # 价格为0
        valid, msg = OrderValidator.validate_price(0, 'LIMIT')
        assert valid is False
    
    def test_validate_order(self):
        """测试完整订单验证"""
        order = Order(
            symbol='000001.SZ',
            side='BUY',
            quantity=100,
            price=10.5,
            order_type='LIMIT',
        )
        
        valid, msg = OrderValidator.validate_order(order)
        assert valid is True


class TestFundChecker:
    """测试资金检查器"""
    
    def test_check_buy_fund_sufficient(self):
        """测试资金充足"""
        valid, msg, need = FundChecker.check_buy_fund(
            price=10.0,
            quantity=100,
            available_cash=2000.0,
        )
        assert valid is True
        assert need > 0
    
    def test_check_buy_fund_insufficient(self):
        """测试资金不足"""
        valid, msg, need = FundChecker.check_buy_fund(
            price=10.0,
            quantity=100,
            available_cash=500.0,
        )
        assert valid is False
        assert "资金不足" in msg
    
    def test_check_sell_position_sufficient(self):
        """测试持仓充足"""
        positions = {
            '000001.SZ': {'quantity': 1000, 'available': 500}
        }
        
        valid, msg = FundChecker.check_sell_position(
            symbol='000001.SZ',
            quantity=100,
            positions=positions,
        )
        assert valid is True
    
    def test_check_sell_position_insufficient(self):
        """测试持仓不足"""
        positions = {
            '000001.SZ': {'quantity': 100, 'available': 50}
        }
        
        valid, msg = FundChecker.check_sell_position(
            symbol='000001.SZ',
            quantity=100,
            positions=positions,
        )
        assert valid is False


class TestTradingTimeChecker:
    """测试交易时间检查器"""
    
    def test_is_trading_day(self):
        """测试交易日判断"""
        # 周一
        dt = datetime(2026, 3, 2)  # 周一
        assert TradingTimeChecker.is_trading_day(dt) is True
        
        # 周日
        dt = datetime(2026, 3, 1)  # 周日
        assert TradingTimeChecker.is_trading_day(dt) is False
    
    def test_is_trading_time(self):
        """测试交易时间判断"""
        # 交易时间 (周一上午10点)
        dt = datetime(2026, 3, 2, 10, 0)
        valid, msg = TradingTimeChecker.is_trading_time(dt)
        assert valid is True
        
        # 非交易时间 (周一早上8点)
        dt = datetime(2026, 3, 2, 8, 0)
        valid, msg = TradingTimeChecker.is_trading_time(dt)
        assert valid is False


class TestTradingSafety:
    """测试交易安全总控"""
    
    def test_safety_check_buy(self):
        """测试买入安全检查"""
        safety = TradingSafety(
            enable_time_check=False,  # 禁用时间检查 (测试用)
            enable_fund_check=True,
            enable_order_validation=True,
            enable_position_check=True,
        )
        
        order = Order(
            symbol='000001.SZ',
            side='BUY',
            quantity=100,
            price=10.0,
            order_type='LIMIT',
        )
        
        # 资金充足
        valid, msg = safety.check_order(
            order,
            available_cash=2000.0,
            positions={},
        )
        assert valid is True
        
        # 资金不足
        valid, msg = safety.check_order(
            order,
            available_cash=500.0,
            positions={},
        )
        assert valid is False
    
    def test_safety_check_sell(self):
        """测试卖出安全检查"""
        safety = TradingSafety(
            enable_time_check=False,
            enable_fund_check=True,
            enable_order_validation=True,
            enable_position_check=True,
        )
        
        order = Order(
            symbol='000001.SZ',
            side='SELL',
            quantity=100,
            order_type='MARKET',
        )
        
        positions = {
            '000001.SZ': {'quantity': 1000, 'available': 500}
        }
        
        # 持仓充足
        valid, msg = safety.check_order(
            order,
            available_cash=0,
            positions=positions,
        )
        assert valid is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
