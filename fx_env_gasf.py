import gym
import gym.spaces
import pandas as pd
import numpy as np
import datetime as dt
from typing import List

import fx_calc
import processing

STAY = 0
BUY = 1
SELL = 2
CLOSE = 3

TECH_SAFE_START_IDX = 170

SPREAD = 0.003
LEVERAGE = 25.0
MARGIN_RATIO = 1.0/LEVERAGE
INIT_BALANCE = 100000.0
PRICE_TO_PIPS = 100
PIPS_TO_PRICE = 0.01

RISK_THRESH = 0.02
LOSS_CUT_PIPS = 20

VOLATILITY_TGT = 5.0


def action2int(action):
    if action == BUY:
        return 1
    if action == SELL:
        return -1
    return 0


def pips2price(pips):
    return pips * PIPS_TO_PRICE


def price2pips(price):
    return price * PRICE_TO_PIPS


class Position():
    def __init__(self, open_price, size, broker):
        self.size = size
        self.open_price = open_price
        self.close_price = -1
        self.broker = broker  # type: Broker
        self.open_time = broker.now_time
        self.close_time = dt.datetime(2000, 1, 1)

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
        return price2pips(self.broker.now_price - self.open_price)

    def close(self):
        self.close_time = self.broker.now_time
        self.close_price = self.broker.now_price
        return self.pl


class Broker():
    def __init__(self, leverage, balance, df, gasf):
        # 現ステップでの時刻
        self.now_time = dt.datetime(2000, 1, 1)
        # 現ステップでのClose値
        self.now_price = 0
        self.prev_price = 0

        # 残高
        self.balance = balance

        # 取引履歴
        self.long_hists = []
        self.short_hists = []
        # 保持ポジション
        self.positions = []  # type: List[Position]
        self.position_size = 0

        # ステップイテレータ
        self.iter = 0
        self.df = df
        self.gasf = gasf

        self.volatility_arr = []
        self.setup_volatility_arr(self.df.Close.values.tolist(), 60)

    def setup_volatility_arr(self, rate_arr, window_size):
        local_window_size = window_size
        for idx in range(len(rate_arr)):
            if idx + 1 < local_window_size:
                self.volatility_arr.append(0)
            else:
                s = (idx + 1) - local_window_size
                tmp_arr = rate_arr[s:idx + 1]
                self.volatility_arr.append(
                    fx_calc.calculate_volatility(tmp_arr, local_window_size))

    @property
    def has_long(self):
        return len(self.positions) > 0 and self.positions[0].is_long

    @property
    def has_short(self):
        return len(self.positions) > 0 and self.positions[0].is_short

    # 最大損失許容量
    @property
    def max_allowable_loss(self):
        return self.balance * RISK_THRESH

    # 取引ポジションサイズ
    @property
    def trade_size(self):
        # return self.balance * RISK_THRESH / pips2price(LOSS_CUT_PIPS)
        return 1000

    # 必要証拠金
    @property
    def margin(self):
        return self.position_size * MARGIN_RATIO

    def get_volatility(self, delta):
        return self.volatility_arr[self.iter + delta]

    def get_gasf_data(self):
        return self.gasf[self.iter]

    # 内部状態を更新する（各ステップの最初に必ず呼び出す)
    # return: is_last
    def update(self):
        self.prev_price = self.now_price
        self.now_price = self.df.Close.iloc[self.iter]
        self.now_time = self.df.Datetime.iloc[self.iter]
        self.iter += 1
        return self.iter >= len(self.df)-1 or self.balance <= 0

    # ロスカットが必要かどうか
    def need_loss_cut(self):
        return self.margin > self.max_allowable_loss

    # ポジションを作成
    def open_position(self, size):
        pos = Position(self.now_price, size, self)
        self.positions.append(pos)
        self.position_size += size

    def buy(self):
        self.open_position(self.trade_size)

    def sell(self):
        self.open_position(-self.trade_size)

    # ポジションを確定する
    def close(self, close_long=True, close_short=True):
        pl = 0
        for pos in self.positions:
            # Long or Shortの条件に合致するなら
            if (pos.is_long and close_long) or (pos.is_short and close_short):
                # ポジションを確定して損益を計上
                pl += pos.close() - SPREAD * pos.size
                self.position_size -= pos.size
                # 取引履歴に追加
                if pos.is_long:
                    self.long_hists.append(pos)
                elif pos.is_short:
                    self.short_hists.append(pos)
                # 保持ポジションリストから削除
                self.positions.remove(pos)
        self.balance += pl


class FxEnv_GASF(gym.Env):
    TEST_MODE = 'test'
    TRAIN_MODE = 'train'

    def __init__(self, df, mode: str):
        self.df = df
        self.gasf = processing.get_ohlc_culr_gasf(df.loc[:, 'Open': 'Close'])
        self.gasf = processing.nwhc2nchw_array(self.gasf)
        self.mode = mode
        self.action_hist = [0, 0, 0]
        self.broker = Broker(LEVERAGE, INIT_BALANCE, self.df, self.gasf)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            # ch, h, w
            shape=(self.gasf.shape[1], self.gasf.shape[2], self.gasf.shape[3], ))
        self.reward_range = (-50.0, 50.0)

    def reset(self):
        self.broker = Broker(MARGIN_RATIO, INIT_BALANCE, self.df, self.gasf)
        for _ in range(TECH_SAFE_START_IDX):
            self.broker.update()
        return self.observe()

    def step(self, action):
        done = self.broker.update()

        # need_loss_cut = self.broker.need_loss_cut()
        if action == STAY:
            pass
        elif action == CLOSE:
            self.broker.close()
        elif action == BUY and not self.broker.has_long:
            self.broker.close()
            self.broker.buy()
        elif action == SELL and not self.broker.has_short:
            self.broker.close()
            self.broker.sell()
        self.action_hist.append(action2int(action))

        return self.observe(), self.calc_rewerd(), done, {}

    def render(self):
        if self.mode == self.TEST_MODE:
            pass
        elif self.mode == self.TRAIN_MODE:
            print('{:>3.1f}% ({}/{})  balance:{:>5.1f}  position:{:>3.1f}                      '.format(self.broker.iter /
                                                                                                        self.broker.gasf.shape[0] * 100, self.broker.iter, self.broker.gasf.shape[0], self.broker.balance, self.broker.position_size), end='\r')

    def observe(self):
        gasf = self.broker.get_gasf_data()
        #gasf = gasf.reshape((1, ) + gasf.shape)
        return gasf

    # 利益計算（https://qiita.com/ryo_grid/items/1552d70eb2a8c15f6fd2 参照）
    def calc_rewerd(self):
        # 取引量（１でいいらしい
        mu = 1
        # 取引コスト
        bp = 0.000015
        # アクション履歴(idx=-1は今回(t)のアクション, (t-1)のアクションはidx=-2)
        A1 = self.action_hist[-2]
        A2 = self.action_hist[-3]
        # ボラティリティ
        sigma1 = self.broker.get_volatility(-1)
        sigma2 = self.broker.get_volatility(-2)
        # 定数でいい？計算式がわからん
        sigma_tgt = VOLATILITY_TGT
        # 価格履歴
        p1 = self.broker.prev_price
        rt = self.broker.now_price - self.broker.prev_price

        diff = A1*(sigma_tgt/sigma1)*rt
        cost = bp*p1*abs((sigma_tgt/sigma1)*A1 - (sigma_tgt/sigma2)*A2)

        return mu*(diff - cost)

    def close(self):
        if mode == self.TEST_MODE:
            # グラフのプロット？
            pass
        elif mode == self.TRAIN_MODE:
            pass
