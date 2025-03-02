from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import gym
import time
from gym import spaces
from numpy import floating
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger  import KVWriter,Logger,configure
import wandb


class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 记录每一步的指标
        for info in self.locals['infos']:
            if 'total_assets' in info:
                wandb.log({
                    "metrics/total_assets": info['total_assets'],
                    "metrics/gain_loss_pct": info['gain_loss_pct'],
                    "metrics/transactions": info['transactions'],
                    "metrics/cash": info['cash'],
                    "global_step": self.num_timesteps,
                }, commit=False)
        return True

    def _on_rollout_end(self) -> None:
        # 记录每个回合的指标
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done and 'terminal_total_assets' in info:
                wandb.log({
                    "episode/terminal_total_assets": info['terminal_total_assets'],
                    "episode/terminal_gain_loss_pct": info['terminal_gain_loss_pct'],
                    "episode/total_trades": info['episode_total_trades'],
                    "episode/episode": self.n_episodes,
                    "episode/termination_reason": info['reason'],
                    "global_step": self.num_timesteps,
                })
        super()._on_rollout_end()


class StockLearningEnv(gym.Env):
    """构建强化学习交易环境

        Attributes
            df: 构建环境时所需要用到的行情数据
            buy_cost_pct: 买股票时的手续费
            sell_cost_pct: 卖股票时的手续费
            date_col_name: 日期列的名称
            hmax: 最大可交易的数量
            print_verbosity: 打印的频率
            initial_amount: 初始资金量
            daily_information_cols: 构建状态时所考虑的列
            cache_indicator_data: 是否把数据放到内存中
            random_start: 是否随机位置开始交易（训练和回测环境分别为True和False）
            patient: 是否在资金不够时不执行交易操作，等到有足够资金时再执行
            currency: 货币单位
    """

    metadata = {"render.modes": ["human"]}
    def __init__(
        self,
        df: pd.DataFrame,
        buy_cost_pct: float = 3e-3,
        sell_cost_pct: float = 3e-3,
        date_col_name: str = "date",
        hmax: int = 10,
        print_verbosity: int = 10,
        initial_amount: int = 1e6,
        daily_information_cols: List = ["open", "close", "high", "low", "volume"],
        cache_indicator_data: bool = True,
        random_start: bool = True,
        patient: bool = False,
        currency: str = "￥",
        logger=None

    ) -> None:
        self.current_transaction = None
        self.df = df
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(date_col_name)
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.daily_information_cols = daily_information_cols
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.turbulence = 0
        self.episode = -1
        self.episode_history = []
        self.printed_header = False
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.max_total_assets = 0
        if self.cache_indicator_data:
            """cashing data 的结构:
               [[date1], [date2], [date3], ...]
               date1 : [stock1 * cols, stock2 * cols, ...]
            """
            print("加载数据缓存")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            print("数据缓存成功!")
            # 初始化 Logger
        if logger is None:
            # 如果没有传入 logger，创建一个默认的（输出到控制台）
            self.logger = Logger(None, [])
        else:
            self.logger = logger
        self.risk_params = {
            'max_leverage': 3.0,  # 最大杠杆倍数
            'sector_limit': 0.3,  # 单一行业持仓限制
            'stop_loss': -0.15  # 最大回撤止损
        }
    def seed(self, seed: Any = None) -> None:
        """设置随机种子"""
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)
    
    @property
    def current_step(self) -> int:
        """当前回合的运行步数"""
        return self.date_index - self.starting_point
    
    @property
    def cash_on_hand(self) -> float:
        """当前拥有的现金"""
        return self.state_memory[-1][0]
    
    @property
    def holdings(self) -> List:
        """当前的持仓数据"""
        return self.state_memory[-1][1: len(self.assets) + 1]

    @property
    def closings(self) -> List:
        """每支股票当前的收盘价"""
        return np.array(self.get_date_vector(self.date_index, cols=["close"]))

    def get_date_vector(self, date: int, cols: List = None) -> List:
        """获取 date 那天的行情数据"""
        if(cols is None) and (self.cached_data is not None):
            return self.cached_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[[date]]
            res = []
            for asset in self.assets:
                tmp_res = trunc_df[trunc_df[self.stock_col] == asset]
                res += tmp_res.loc[date, cols].tolist()
            assert len(res) == len(self.assets) * len(cols)
            return res
    
    def reset(self) -> np.ndarray:
        self.seed()
        self.sum_trades = 0
        self.max_total_assets = self.initial_amount
        if self.random_start:
            self.starting_point = random.choice(range(int(len(self.dates) * 0.5)))
        else:
            self.starting_point = 0
        self.date_index = self.starting_point
        self.turbulence = 0
        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": []
        }
        init_state = np.array(
            [self.initial_amount] 
            + [0] * len(self.assets)
            + self.get_date_vector(self.date_index)
        )
        self.state_memory.append(init_state)
        return init_state

    def log_step(
        self, reason: str, terminal_reward: float=None
        ) -> None:
        """打印"""
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
        
        assets = self.account_information["total_assets"][-1]
        tmp_retreat_ptc = assets / self.max_total_assets - 1
        retreat_pct = tmp_retreat_ptc if assets < self.max_total_assets else 0
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount

        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(assets))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{retreat_pct*100:0.2f}%"
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def return_terminal(
            self, reason: str = "Last Date", reward: int = 0
    ) -> Tuple[List, float, bool, dict]:
        """terminal 时执行的操作"""
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount

        # 使用 self.logger 替代 logger.Logger
        self.logger.record("environment/GainLoss_pct", (gl_pct - 1) * 100)
        self.logger.record(
            "environment/total_assets",
            int(self.account_information["total_assets"][-1])
        )
        reward_pct = gl_pct
        self.logger.record("environment/total_reward_pct", (reward_pct - 1) * 100)
        self.logger.record("environment/total_trades", self.sum_trades)

        # 确保 current_step 不为零
        if self.current_step == 0:
            avg_daily_trades = 0.0
        else:
            avg_daily_trades = self.sum_trades / self.current_step

        self.logger.record("environment/avg_daily_trades", avg_daily_trades)
        self.logger.record(
            "environment/avg_daily_trades_per_asset",
            avg_daily_trades / len(self.assets) if self.assets.size > 0 else 0.0
        )
        self.logger.record("environment/completed_steps", self.current_step)
        self.logger.record(
            "environment/sum_rewards",
            np.sum(self.account_information["reward"])
        )
        self.logger.record(
            "environment/retreat_proportion",
            self.account_information["total_assets"][-1] / self.max_total_assets
            if self.max_total_assets != 0 else 0.0
        )
        # 记录终止时的关键指标
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount - 1
        info = {
            "terminal_total_assets": self.account_information["total_assets"][-1],
            "terminal_gain_loss_pct": gl_pct * 100,
            "episode_total_trades": self.sum_trades,
            "episode": self.episode,
            "reason": reason,
        }

        return state, reward, True, info

    def log_header(self) -> None:
        """Log 的列名"""
        if not self.printed_header:
            self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"
            # 0, 1, 2, ... 是序号
            # 4, 4, 15, ... 是占位格的大小
            print(
                self.template.format(
                    "EPISODE",
                    "STEPS",
                    "TERMINAL_REASON",
                    "CASH",
                    "TOT_ASSETS",
                    "TERMINAL_REWARD",
                    "GAINLOSS_PCT",
                    "RETREAT_PROPORTION"
                )
            )
            self.printed_header = True

    def calculate_transaction_cost(self) -> float:
        """综合计算交易成本（含手续费+滑点+市场冲击）"""
        # 获取当前交易方向和数量
        transactions = self.current_transaction  # shape: (n_assets,)
        current_prices = self.closings  # 当前各资产价格

        # 计算基础手续费
        buy_mask = transactions > 0
        sell_mask = transactions < 0
        buy_volume = np.sum(transactions[buy_mask] * current_prices[buy_mask])
        sell_volume = np.sum(-transactions[sell_mask] * current_prices[sell_mask])

        # 基础手续费模型
        base_cost = buy_volume * self.buy_cost_pct + sell_volume * self.sell_cost_pct

        # 滑点模型（随机+成交量比例）
        slippage_factor = 0.001  # 基础滑点率
        slippage = (
                np.random.normal(loc=0.5, scale=0.2) * slippage_factor *
                np.abs(transactions) * current_prices
        )
        total_slippage = np.sum(slippage)

        # 市场冲击模型（考虑交易量占总成交比例）
        avg_daily_volume = np.array([  # 需预先计算各资产日均成交量
            self.df[self.df['tic'] == asset]['volume'].mean()
            for asset in self.assets
        ])
        volume_ratio = np.clip(
            np.abs(transactions) / (avg_daily_volume + 1e-6),
            0, 0.3
        )  # 限制在30%以内
        impact_cost = 0.1 * volume_ratio ** 2 * current_prices * np.abs(transactions)

        return float(base_cost + total_slippage + np.sum(impact_cost))

    def calculate_risk(self) -> dict:
        asset_values = np.array(self.account_information["total_assets"])
        holdings = np.array(self.holdings)
        current_prices = self.closings

        risk_metrics = {}

        # ===== 波动率计算改进 =====
        min_window = 5  # 最小计算窗口
        if len(asset_values) > min_window:
            returns = np.diff(asset_values) / asset_values[:-1]
            # 使用滚动窗口+衰减系数
            decay_factor = 0.94  # 对应RiskMetrics的lambda
            volatility = 0
            for r in returns[::-1]:
                volatility = decay_factor * volatility + (1 - decay_factor) * r ** 2
            risk_metrics['volatility'] = np.sqrt(volatility * 252)  # 年化
        else:
            risk_metrics['volatility'] = 0.3  # 默认值

        # ===== 最大回撤改进 =====
        if len(asset_values) > 0:
            peak = np.maximum.accumulate(asset_values)
            trough = np.minimum.accumulate(asset_values[::-1])[::-1]  # 未来最低点
            risk_metrics['max_drawdown'] = np.nanmax((peak - trough) / peak)
        else:
            risk_metrics['max_drawdown'] = 0

        # ===== 持仓集中度改进 =====
        position_values = holdings * current_prices
        total_position = np.sum(position_values)
        if total_position > 1e-6:  # 有效持仓时才计算
            herfindahl = np.sum((position_values / total_position) ** 2)
            risk_metrics['concentration'] = herfindahl * 100
        else:
            risk_metrics['concentration'] = 0  # 零持仓时无风险

        # ===== 流动性风险评估 =====
        avg_volume = np.mean([v for v in self.get_date_vector(
            self.date_index, cols=["volume_20_sma"]
        ) if v > 0])
        risk_metrics['liquidity_risk'] = np.log1p(total_position / (avg_volume + 1e-6))

        return risk_metrics

    def get_reward(self) -> float:
        current_assets = self.account_information["total_assets"][-1]
        risk_metrics = self.calculate_risk()

        # ===== 核心改进点 =====
        # 1. 动态基准收益率（考虑存活偏差）
        alive_days = len(self.account_information["total_assets"])
        market_benchmark = (1.08 ** (alive_days / 252) - 1)  # 年化8%作为基准

        # 2. 收益计算（相对基准）
        excess_return = (current_assets / self.initial_amount - 1) - market_benchmark

        # 3. 风险调整（改进夏普计算）
        volatility = np.clip(risk_metrics['volatility'], 0.15, 0.6)  # 限制波动率范围
        sharpe_ratio = excess_return / (volatility + 1e-6)

        # 4. 持仓健康度（考虑零持仓情况）
        holdings = np.array(self.holdings) + 1e-6  # 防止零除
        position_weights = holdings / (np.sum(np.abs(holdings)) + 1e-6)
        strategy_entropy = -np.sum(position_weights * np.log(position_weights + 1e-6))

        # 5. 交易频率惩罚（抑制无效交易）
        trade_frequency = len(self.transaction_memory) / (self.current_step + 1)
        trade_penalty = 0.05 * np.tanh(trade_frequency / 0.2)  # 柔性惩罚

        # ===== 复合奖励 =====
        reward = (
                0.5 * sharpe_ratio +
                0.3 * strategy_entropy -
                0.1 * risk_metrics['concentration'] -
                0.05 * self.calculate_transaction_cost() / current_assets -
                trade_penalty
        )

        # ===== 边界处理 =====
        clipped_reward = np.clip(reward, -5, 5)

        # 记录分量到字典
        self.reward_components = {
            "total": clipped_reward,
            "sharpe": 0.5 * sharpe_ratio,
            "entropy": 0.3 * strategy_entropy,
            "concentration": -0.1 * risk_metrics['concentration'],
            "cost": -0.05 * self.calculate_transaction_cost() / current_assets,
            "trade_penalty": -trade_penalty
        }

        return clipped_reward
    # action 给出动作  乘以   self.hmax 来决定买多少
    def get_transactions(self, actions: np.ndarray) -> np.ndarray:
        """获取实际交易的股数"""
        self.actions_memory.append(actions)
        actions = actions * self.hmax

        # 收盘价为 0 的不进行交易
        actions = np.where(self.closings > 0, actions, 0)

        # 去除被除数为 0 的警告
        out = np.zeros_like(actions)
        zero_or_not = self.closings != 0
        actions = np.divide(actions, self.closings, out=out, where = zero_or_not)
        
        # 不能卖的比持仓的多
        actions = np.maximum(actions, -np.array(self.holdings))

        # 将 -0 的值全部置为 0
        actions[actions == -0] = 0

        return actions

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List, float, bool, dict]:
        transactions = self.get_transactions(actions)
        self.current_transaction = transactions
        self.sum_trades += np.sum(np.abs(actions))
        self.log_header()
        if(self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        if self.date_index == len(self.dates) - 1:
            return self.return_terminal(reward=self.get_reward())
        else:
            begin_cash = self.cash_on_hand
            assert min(self.holdings) >= 0
            assert_value = np.dot(self.holdings, self.closings) # dot 点积算所有持有股票价值
            self.account_information["cash"].append(begin_cash)
            self.account_information["asset_value"].append(assert_value)
            self.account_information["total_assets"].append(begin_cash + assert_value)
            reward = self.get_reward()
            self.account_information["reward"].append(reward)


            sells = -np.clip(transactions, -np.inf, 0)# ​提取交易中的卖出部分
            proceeds = np.dot(sells, self.closings)
            costs = proceeds * self.sell_cost_pct
            coh = begin_cash + proceeds # 计算现金的数量

            buys = np.clip(transactions, 0, np.inf)
            spend = np.dot(buys, self.closings)
            costs += spend * self.buy_cost_pct

            if (spend + costs) > coh: # 如果买不起
                if self.patient:
#                     self.log_step(reason="CASH SHORTAGE")
                    transactions = np.where(transactions > 0, 0, transactions)
                    spend = 0
                    costs = 0
                else:
                    return self.return_terminal(
                        reason="CASH SHORTAGE", reward=self.get_reward()
                    )
            self.transaction_memory.append(transactions)
            assert (spend + costs) <= coh
            coh = coh - spend - costs
            holdings_updated = self.holdings + transactions
            self.date_index += 1
            state = (
                [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(state)

            # 在正常step中返回的info添加指标
            current_assets = begin_cash + assert_value
            gl_pct = (current_assets / self.initial_amount) - 1
            info = {
                "total_assets": current_assets,
                "gain_loss_pct": gl_pct * 100,
                "transactions": np.sum(np.abs(transactions)),
                "cash": coh,
                "step": self.current_step,
            }

            return state, reward, False, info

    def get_sb_env(self) -> Tuple[Any, Any]:
        def get_self():
            return deepcopy(self)
        
        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(
        self, n: int = 10
    ) -> Tuple[Any, Any]:
        def get_self():
            return deepcopy(self)
        
        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs
    
    def save_asset_memory(self) -> pd.DataFrame:
        if self.current_step == 0:
            return None
        else:
            self.account_information["date"] = self.dates[
                -len(self.account_information["cash"]):
            ]
            return pd.DataFrame(self.account_information)
    
    def save_action_memory(self) -> pd.DataFrame:
        if self.current_step == 0:
            return None
        else:
            return pd.DataFrame(
                {
                    "date": self.dates[-len(self.account_information["cash"]):],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory
                }
            )