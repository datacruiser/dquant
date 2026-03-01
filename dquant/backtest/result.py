"""
回测结果
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from dquant.backtest.portfolio import Portfolio
from dquant.backtest.metrics import Metrics


@dataclass
class BacktestResult:
    """回测结果"""
    
    portfolio: Portfolio
    trades: pd.DataFrame
    metrics: Metrics
        
    def __repr__(self):
        return f"BacktestResult(sharpe={self.metrics.sharpe:.2f}, return={self.metrics.total_return:.2%})"
    
    def plot(self, kind: str = 'nav', save_path: Optional[str] = None):
        """
        绘制回测结果
        
        Args:
            kind: 图表类型
                - 'nav': 净值曲线
                - 'returns': 收益率分布
                - 'drawdown': 回撤曲线
                - 'trades': 交易分布
            save_path: 保存路径 (可选)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if kind == 'nav':
                # 净值曲线
                nav = self.portfolio.nav
                ax.plot(nav.index, nav.values, label='策略净值', linewidth=2)
                ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
                ax.set_title('策略净值曲线', fontsize=14, fontweight='bold')
                ax.set_xlabel('日期')
                ax.set_ylabel('净值')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif kind == 'returns':
                # 收益率分布
                returns = self.portfolio.nav.pct_change().dropna()
                ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(returns.mean(), color='red', linestyle='--', 
                          label=f'均值: {returns.mean():.2%}')
                ax.set_title('日收益率分布', fontsize=14, fontweight='bold')
                ax.set_xlabel('收益率')
                ax.set_ylabel('频数')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif kind == 'drawdown':
                # 回撤曲线
                nav = self.portfolio.nav
                running_max = nav.cummax()
                drawdown = (nav - running_max) / running_max
                ax.fill_between(drawdown.index, drawdown.values, 0, 
                               alpha=0.3, color='red', label='回撤')
                ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
                ax.set_title('策略回撤', fontsize=14, fontweight='bold')
                ax.set_xlabel('日期')
                ax.set_ylabel('回撤')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif kind == 'trades':
                # 交易分布
                if len(self.trades) > 0:
                    trade_pnl = self.trades['pnl'] if 'pnl' in self.trades.columns else pd.Series()
                    if len(trade_pnl) > 0:
                        ax.hist(trade_pnl, bins=30, alpha=0.7, edgecolor='black')
                        ax.axvline(trade_pnl.mean(), color='red', linestyle='--',
                                  label=f'均值: {trade_pnl.mean():.2f}')
                        ax.set_title('交易盈亏分布', fontsize=14, fontweight='bold')
                        ax.set_xlabel('盈亏')
                        ax.set_ylabel('频数')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, '无交易数据', ha='center', va='center')
            
            # 格式化日期
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存或显示
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图表已保存: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("警告: matplotlib 未安装，无法绑图")
        except Exception as e:
            print(f"绘图错误: {e}")
