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
# 所持金の1%を賭ける
TRADE_RATIO = 0.01
# 2%で損切り　
LOSS_CUT_RATIO = 0.02
# 最大ポジションユニット数割合
MAX_POSITION_UNIT_RATIO = 0.01
# 最大ポジション超え罰則割合(資金の5%くらい)
OVER_POSITION_DEMERIT = -1

PL_HIST_LENGTH = 10


class Position():
    def __init__(self, units: int, open_pri: float):
        self.open_price = open_pri
        self.units = units

    def get_pl(self, close_pri: float):
        pips = close_pri - self.open_price
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

    def take_pl(self, pl):
        self.balance += pl


class FxEnv(gym.Env):
    def __init__(self, tech_df, scaler):
        # 定数の定義
        self.STAY = 0
        self.BUY = 1
        self.SELL = 2
        self.CLOSE = 3

        # 学習に関する設定
        self.window_size = 5  # 過去何本分のロウソクを見るか

        self.df = tech_df
        data = tech_df.values[:, 1:]
        # データの正規化
        self.data = scaler.transform(data)

        self.data_iter = 0

        # 初期値の定義
        self.init_balance = 1000
        self.pl_hist = []

        # 環境の設定
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.window_size*self.data.shape[1]+2, ))
        return

    @property
    def pl_rate(self):
        profit = 0
        loss = 0
        for pl in self.pl_hist:
            if pl > 0:
                profit += pl
            else:
                loss += - pl
        if loss == 0:
            return 1000000
        return profit/loss*100

    @property
    def now_price(self):
        return self.df.iloc[self.data_iter]["Close"]

    @property
    def done(self):
        # データの終端で終了
        return (self.data_iter + 1 >= len(self.data)) or self.account.balance <= 0

    @property
    def trade_units(self):
        return self.account.balance * TRADE_RATIO

    def reset(self):
        self.account = Account(self.init_balance)
        self.data_iter = self.window_size - 1

        return self._observe()

    def step(self, action):
        reward = 0
        done = self.done
        unrealized_pl = self.account.get_unrealized_pl(self.now_price)

        # ロスカット判定
        is_loss_cut = self.account.balance + \
            unrealized_pl < self.account.balance * LOSS_CUT_RATIO

        # アクションに応じて行動
        if action == self.CLOSE and self.account.position_units == 0:
            reward -= 1
        if action == self.CLOSE or done or is_loss_cut:
            reward += self.close()
        elif action == self.STAY:
            pass
        elif abs(self.account.position_units) + self.trade_units > self.account.balance * MAX_POSITION_UNIT_RATIO:
            # ポジション上限を超えそうなら
            # ペナルティを課して取引はしない
            reward += OVER_POSITION_DEMERIT
        elif action == self.BUY:
            reward += self.buy()
        elif action == self.SELL:
            reward += self.sell()

        self.data_iter += 1

        return self._observe(), reward, self.done, self._info()

    def render(self):
        print('{:>3.1f}% ({}/{})  balance:{:>5.1f}  position:{:>3.1f}  損益率:{:>3.1f}　　　　'.format(self.data_iter/len(self.data)
                                                                                                * 100, self.data_iter, len(self.data), self.account.balance, self.account.position_units, self.pl_rate), end='\r')
        return

    def _observe(self):
        return np.append(self.data[self.data_iter - self.window_size + 1: self.data_iter + 1].ravel(),
                         [self.account.position_units, self.account.get_unrealized_pl(self.now_price)/self.account.balance])

    def _info(self):
        return {'balance': self.account.balance, 'datetime': self.df.iloc[self.data_iter]['Datetime']}

    def update_pips_stat(self, new_pips):
        self.pl_hist.append(new_pips)

    # すべてのポジションを閉じて損益を返す
    def close(self, close_long=True, close_short=True):
        pl = 0
        for pos in self.account.positions:
            # long/shortが指定通りならクローズ
            if (close_long and pos.is_long) or (close_short and not pos.is_long):
                new_pips, new_pl = pos.get_pl(self.now_price)
                pl += new_pl
                self.update_pips_stat(new_pips)
                self.account.positions.remove(pos)

        # 内部変数を更新
        self.account.take_pl(pl)
        return pl

    # 買い注文を出し，損益を返す
    def buy(self):
        # shortポジションをクローズ
        pl = self.close(close_long=False, close_short=True)
        # 買いポジションを追加
        self.account.positions.append(
            Position(units=self.trade_units, open_pri=self.now_price))
        return pl

    def sell(self):
        # longポジションをクローズ
        pl = self.close(close_long=True, close_short=False)
        # 売りポジションを追加
        self.account.positions.append(
            Position(units=-self.trade_units, open_pri=self.now_price))
        return pl
