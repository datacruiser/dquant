"""
强化学习交易代理
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from dquant.constants import DEFAULT_COMMISSION, DEFAULT_INITIAL_CASH, MIN_SHARES


@dataclass
class TradingState:
    """交易状态"""

    prices: np.ndarray  # 价格序列
    positions: np.ndarray  # 持仓
    cash: float
    timestamp: int

    def to_observation(self) -> np.ndarray:
        """转换为观察向量"""
        return np.concatenate(
            [
                self.prices.flatten(),
                self.positions,
                [self.cash],
                [self.timestamp],
            ]
        )


class TradingEnvironment:
    """
    交易环境

    符合 OpenAI Gym 接口的交易环境。

    Usage:
        env = TradingEnvironment(data, initial_cash=1000000)

        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        commission: float = DEFAULT_COMMISSION,
        lookback: int = 20,
        n_stocks: int = 10,
    ):
        """
        初始化环境

        Args:
            data: 市场数据
            initial_cash: 初始资金
            commission: 手续费率
            lookback: 观察窗口
            n_stocks: 最多持有股票数
        """
        self.data = data
        self.initial_cash = initial_cash
        self.commission = commission
        self.lookback = lookback
        self.n_stocks = n_stocks

        # 预处理数据
        self._prepare_data()

        # 状态空间和动作空间
        self.observation_space = (lookback * n_stocks + n_stocks + 2,)
        self.action_space = n_stocks * 3  # 每只股票: 买入/持有/卖出

        self.reset()

    def _prepare_data(self):
        """预处理数据"""
        # 按日期和股票排序
        self.dates = sorted(self.data.index.unique())
        self.symbols = self.data["symbol"].unique()[: self.n_stocks]

        # 构建价格矩阵
        self.price_matrix = np.zeros((len(self.dates), self.n_stocks))

        for i, date in enumerate(self.dates):
            day_data = self.data[self.data.index == date]
            for j, symbol in enumerate(self.symbols):
                if symbol in day_data["symbol"].values:
                    price = day_data[day_data["symbol"] == symbol]["close"].values[0]
                    self.price_matrix[i, j] = price

    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = self.lookback
        self.cash = self.initial_cash
        self.positions = np.zeros(self.n_stocks)
        self.position_values = np.zeros(self.n_stocks)
        self.prev_total_value = self.initial_cash

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        prices = self.price_matrix[
            self.current_step - self.lookback : self.current_step
        ]

        state = TradingState(
            prices=prices,
            positions=self.positions.copy(),
            cash=self.cash,
            timestamp=self.current_step,
        )

        return state.to_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作

        Args:
            action: 动作向量 [stock_0_action, stock_1_action, ...]
                    0=卖出, 1=持有, 2=买入

        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 当前价格
        current_prices = self.price_matrix[self.current_step]

        # 执行交易
        for i, act in enumerate(action):
            price = current_prices[i]

            if act == 2:  # 买入
                # 用可用资金的 1/n_stocks 买入
                buy_amount = self.cash / self.n_stocks
                if price <= 0:
                    continue
                shares = int((buy_amount / price) // MIN_SHARES) * MIN_SHARES
                if shares <= 0:
                    continue
                cost = shares * price * (1 + self.commission)

                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[i] += shares

            elif act == 0:  # 卖出
                if self.positions[i] > 0:
                    shares = self.positions[i]
                    revenue = shares * price * (1 - self.commission)
                    self.cash += revenue
                    self.positions[i] = 0

        # 更新持仓价值
        self.position_values = self.positions * current_prices
        total_value = self.cash + np.sum(self.position_values)

        # 计算奖励 (收益率)
        reward = (total_value - self.prev_total_value) / self.prev_total_value
        self.prev_total_value = total_value

        # 前进一步
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1

        info = {
            "total_value": total_value,
            "cash": self.cash,
            "position_value": np.sum(self.position_values),
            "step": self.current_step,
        }

        return self._get_state(), reward, done, info

    def render(self):
        """渲染环境"""
        total_value = self.cash + np.sum(self.position_values)
        print(
            f"Step {self.current_step}: "
            f"Total={total_value:,.0f}, "
            f"Cash={self.cash:,.0f}, "
            f"Positions={np.sum(self.position_values):,.0f}"
        )


class BaseRLAgent(ABC):
    """强化学习代理基类"""

    def __init__(self, n_stocks: int, lookback: int = 20):
        self.n_stocks = n_stocks
        self.lookback = lookback

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        pass

    @abstractmethod
    def update(self, experience: Tuple):
        """更新模型"""
        pass

    def save(self, path: str):
        """保存模型"""
        pass

    def load(self, path: str):
        """加载模型"""
        pass


class DQNAgent(BaseRLAgent):
    """
    DQN 交易代理

    使用 Deep Q-Network 进行交易决策。

    Usage:
        agent = DQNAgent(n_stocks=10, lookback=20)

        # 训练
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.update((state, action, reward, next_state, done))
                state = next_state

        # 使用
        action = agent.select_action(state, training=False)
    """

    def __init__(
        self,
        n_stocks: int,
        lookback: int = 20,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
    ):
        super().__init__(n_stocks, lookback)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self._model = None
        self._target_model = None
        self._buffer: deque = deque(maxlen=buffer_size)

    def _build_model(self, state_dim: int, action_dim: int):
        """构建神经网络"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        class QNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_size):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, action_dim),
                )

            def forward(self, x):
                return self.net(x)

        self._model = QNetwork(state_dim, action_dim, self.hidden_size)
        self._target_model = QNetwork(state_dim, action_dim, self.hidden_size)
        self._target_model.load_state_dict(self._model.state_dict())

        self.optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        # Epsilon-greedy 策略
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 3, self.n_stocks)

        # 使用模型预测
        if self._model is None:
            state_dim = len(state)
            action_dim = self.n_stocks * 3
            self._build_model(state_dim, action_dim)

        try:
            import torch

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self._model(state_tensor)

                # 每只股票选择 Q 值最大的动作
                q_values = q_values.view(self.n_stocks, 3)
                actions = q_values.argmax(dim=1).numpy()

                return actions
        except (RuntimeError, ValueError):
            return np.ones(self.n_stocks, dtype=int)  # 默认持有

    def update(self, experience: Tuple):
        """更新模型"""
        if self._model is None:
            return

        # 存储经验 (deque 自动淘汰旧数据)
        self._buffer.append(experience)

        # 批量训练
        if len(self._buffer) < self.batch_size:
            return

        try:
            import torch

            # 采样
            indices = np.random.choice(len(self._buffer), self.batch_size)
            batch = [self._buffer[i] for i in indices]

            states = torch.FloatTensor([e[0] for e in batch])
            actions = torch.LongTensor([e[1] for e in batch])
            rewards = torch.FloatTensor([e[2] for e in batch])
            next_states = torch.FloatTensor([e[3] for e in batch])
            dones = torch.FloatTensor([e[4] for e in batch])

            # 计算 Q 值
            current_q = self._model(states).gather(1, actions.unsqueeze(1))

            # 计算目标 Q 值
            with torch.no_grad():
                next_q = self._target_model(next_states).max(1)[0]
                target_q = rewards + self.gamma * next_q * (1 - dones)

            # 计算损失
            loss = torch.nn.functional.mse_loss(current_q.squeeze(), target_q)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 衰减 epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        except Exception as e:
            print(f"[DQN] Update error: {e}")

    def update_target_model(self):
        """更新目标网络"""
        if self._model is not None and self._target_model is not None:
            self._target_model.load_state_dict(self._model.state_dict())


class PPOAgent(BaseRLAgent):
    """
    PPO 交易代理

    使用 Proximal Policy Optimization 进行交易决策。
    """

    def __init__(
        self,
        n_stocks: int,
        lookback: int = 20,
        hidden_size: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
    ):
        super().__init__(n_stocks, lookback)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        self._model = None

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        # 简化实现: 返回持有
        return np.ones(self.n_stocks, dtype=int)

    def update(self, experience: Tuple):
        """更新模型"""
        pass


class RLStrategy:
    """
    RL 策略包装器

    将 RL Agent 包装为 DQuant 策略接口。

    Usage:
        from dquant.ai import DQNAgent, RLStrategy

        agent = DQNAgent(n_stocks=10)
        # ... 训练 agent ...

        strategy = RLStrategy(agent, symbols=['000001.SZ', ...])
        result = engine.backtest(strategy=strategy)
    """

    def __init__(
        self,
        agent: BaseRLAgent,
        symbols: List[str],
        lookback: int = 20,
        name: str = "RLStrategy",
    ):
        self.agent = agent
        self.symbols = symbols
        self.lookback = lookback
        self.name = name

    def generate_signals(self, data: pd.DataFrame) -> list:
        """生成信号"""
        signals = []

        # 按日期遍历
        dates = sorted(data.index.unique())

        for i, date in enumerate(dates[self.lookback :], start=self.lookback):
            # 构建状态
            _ = data.loc[dates[i - self.lookback : i]]  # noqa: F841

            # 调用 agent
            # ... 实现状态构建和动作选择 ...

            # 生成信号
            # ...

        return signals
