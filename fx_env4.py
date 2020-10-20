import gym
import gym.spaces
import pandas as pd
import numpy as np
import datetime as dt
from typing import List

import fx_calc

# df.max().values[1:].max()
TECH_FILE_VALUE_MAX = 130.0
# df.min().values[1:].min()
TECH_FILE_VALUE_MIN = -6.0

# FX取引に関する定数
# 所持金の0.1%を賭ける
TRADE_RATIO = 0.001
# 2%で損切り　
LOSS_CUT_RATIO = 0.02
# 最大ポジションユニット数割合
MAX_POSITION_UNIT_RATIO = 0.005
# 最大ポジション超え罰則割合(資金の0.01%くらい)
# PENALTY_RATIO = -0.0001
PENALTY_RATIO = 0

PL_HIST_LENGTH = 10


class Position():
    def __init__(self, units: int, open_pri: float):
        self.open_price = open_pri
        self.units = units

    def get_pl(self, close_pri: float):
        pips = (close_pri - self.open_price) * (1 if self.is_long else -1)
        pl = fx_calc.calc_pl(self.open_price, close_pri, self.units)
        return pips, pl

    @property
    def is_long(self):
        return self.units > 0


class Account():
    def __init__(self, balance: float):
        self.balance = balance  # type: float
        self.positions = []  # type: List[Position]
        self.position_sum = 0

    @property
    def has_position(self):
        return len(self.positions) != 0

    @property
    def position_units(self):
        self.position_sum = 0
        for pos in self.positions:
            self.position_sum += pos.units
        return self.position_sum

    def get_unrealized_pl(self, close_price):
        unreal_pl = 0
        for pos in self.positions:
            _, pl = pos.get_pl(close_price)
            unreal_pl += pl
        return unreal_pl

    def get_unrealized_pips(self, close_price):
        unreal_pips = 0
        for pos in self.positions:
            pips, _ = pos.get_pl(close_price)
            unreal_pips += pips
        return unreal_pips

    def take_pl(self, pl):
        self.balance += pl


class FxEnv(gym.Env):
    def __init__(self, tech_df, scaler):
        # 定数の定義
        self.CLOSE = 0
        self.BUY = 1
        self.SELL = 2

        # 学習に関する設定
        self.window_size = 5  # 過去何本分のロウソクを見るか

        self.df = tech_df
        data = tech_df.values[:, 1:]
        # データの正規化
        self.data = scaler.transform(data)

        self.data_iter = 0

        # 初期値の定義
        self.init_balance = 1000
        self.profits = []
        self.losses = []

        # 環境の設定
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.window_size*self.data.shape[1]+2, ))
        return

    @property
    def pl_rate(self):
        profit = np.average(self.profits)
        loss = np.average(self.losses)
        return profit/loss*100

    @property
    def now_price(self):
        return self.df.iat[self.data_iter, 4]

    @property
    def done(self):
        # データの終端で終了
        return (self.data_iter + 2 >= len(self.data)) or self.account.balance <= self.init_balance / 2

    @property
    def trade_units(self):
        return self.account.balance * MAX_POSITION_UNIT_RATIO

    def reset(self):
        self.account = Account(self.init_balance)
        self.data_iter = self.window_size - 1
        self.profits.clear()
        self.losses.clear()

        return self._observe()

    def step(self, action):
        self.data_iter += 1

        reward = 0
        unrealized_pl = self.account.get_unrealized_pl(self.now_price)

        # ロスカット判定
        is_loss_cut = unrealized_pl / self.account.balance < -LOSS_CUT_RATIO

        done = self.done
        # アクションに応じて行動
        # ポジションの保持無しでクローズしたらペナルティ
        if action == self.CLOSE and self.account.position_units == 0:
            reward += PENALTY_RATIO
        if action == self.CLOSE or done or is_loss_cut:
            reward += self.close()
        elif ((action == self.BUY and self.account.position_units > 0) or
              (action == self.SELL and self.account.position_units < 0)):
            # ポジション上限を超えそうなら取引はしない
            pass
        elif action == self.BUY:
            reward += self.buy()
        elif action == self.SELL:
            reward += self.sell()

        return self._observe(), reward, done, self._info()

    def render(self):
        print('{:>3.1f}% ({}/{})  balance:{:>5.1f}  position:{:>3.1f}  損益率:{:>3.1f}　　　　'.format(self.data_iter/len(self.data)
                                                                                                * 100, self.data_iter, len(self.data), self.account.balance, self.account.position_units, self.pl_rate), end='\r')
        return

    def _observe(self):
        return np.append(self.data[self.data_iter - self.window_size + 1: self.data_iter + 1].ravel(),
                         [self.account.position_units/self.trade_units, self.account.get_unrealized_pips(self.now_price)])

    def _info(self):
        return {'balance': self.account.balance, 'datetime': self.df.iat[self.data_iter, 0]}

    def update_pips_stat(self, new_pips):
        if new_pips < 0:
            self.losses.append(-new_pips)
        elif new_pips > 0:
            self.profits.append(new_pips)

    # すべてのポジションを閉じて損益を返す
    def close(self, close_long=True, close_short=True):
        pl = 0
        pips = 0
        for pos in self.account.positions:
            # long/shortが指定通りならクローズ
            if (close_long and pos.is_long) or (close_short and not pos.is_long):
                new_pips, new_pl = pos.get_pl(self.now_price)
                pl += new_pl
                pips += new_pips
                self.update_pips_stat(new_pips)
                self.account.positions.remove(pos)

        # 内部変数を更新
        self.account.take_pl(pl)
        return pips

    # 買い注文を出し，損益を返す
    def buy(self):
        # shortポジションをクローズ
        pips = self.close(close_long=False, close_short=True)
        # 買いポジションを追加
        self.account.positions.append(
            Position(units=self.trade_units, open_pri=self.now_price))
        return pips

    def sell(self):
        # longポジションをクローズ
        pips = self.close(close_long=True, close_short=False)
        # 売りポジションを追加
        self.account.positions.append(
            Position(units=-self.trade_units, open_pri=self.now_price))
        return pips
