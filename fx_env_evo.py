import gym
import gym.spaces
import pandas as pd
import numpy as np
import datetime as dt
from typing import List

STAY = 0
BUY = 1
SELL = 2

MARGIN = 0.04
INIT_BALANCE = 1000.0
PRICE_TO_PIPS = 100


class Position():
    def __init__(self, open_price, size, broker: Broker):
        self.size = size
        self.open_price = open_price
        self.broker = broker  # type: Broker
        self.open_time = broker.now_time
        self.close_time = dt.datetime()

    @property
    def is_long(self):
        return self.size > 0

    @property
    def is_short(self):
        return self.size < 0

    @property
    def pl(self):
        return (self.broker.now_price - self.open_price) * self.size

    @property
    def pips(self):
        return (self.broker.now_price - self.open_price) * PRICE_TO_PIPS

    def close(self):
        self.close_time = self.broker.now_time
        return self.pl


class Broker():
    def __init__(self, margin, balance, df):
        # 現ステップでの時刻
        self.now_time = dt.datetime()
        # 現ステップでのClose値
        self.now_price = 0

        # 必要証拠金割合
        self.margin = margin
        self.leverage = 1 / margin

        # 残高
        self.balance = balance

        # 取引履歴
        self.long_hists = []
        self.short_hists = []
        # 保持ポジション
        self.positions = []  # type: List[Position]

        # ステップイテレータ
        self.iter = 0
        self.df = df

    # 内部状態を更新する（各ステップの最初に必ず呼び出す)
    def update(self):
        self.now_price = self.df.Close[self.iter]
        self.now_time = self.df.Datetime[self.iter]
        self.iter += 1

    # ポジションを作成
    def open_position(self, price, size):
        pos = Position(price, size, self)
        self.positions.append(pos)

    # ポジションを確定する
    def close(self, close_long=True, close_short=True):
        pl = 0
        for pos in self.positions:
            # Long or Shortの条件に合致するなら
            if (pos.is_long and close_long) or (pos.is_short and close_short):
                # ポジションを確定して損益を計上
                pl += pos.close()
                # 取引履歴に追加
                if pos.is_long:
                    self.long_hists.append(pos)
                elif pos.is_short:
                    self.short_hists.append(pos)
                # 保持ポジションリストから削除
                self.positions.remove(pos)


class FxEnv(gym.Env):
    TEST_MODE = 'test'
    TRAIN_MODE = 'train'

    def __init__(self, tech_df, scaler, mode: str):
        self.scaler = scaler
        self.df = tech_df
        self.iter = 0
        self.broker = Broker(MARGIN, INIT_BALANCE)
        self.mode = mode

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        if mode == self.TEST_MODE:
            pass
        elif mode == self.TRAIN_MODE:
            pass

    def observe(self):
        pass

    def close(self):
        if mode == self.TEST_MODE:
            # グラフのプロット？
            pass
        elif mode == self.TRAIN_MODE:
            pass
