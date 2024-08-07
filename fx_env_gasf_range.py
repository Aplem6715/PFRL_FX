import gym
import gym.spaces
import pandas as pd
import numpy as np
import datetime as dt
from typing import List

import fx_calc
import processing
import random

BUY = 1
SELL = 2
CLOSE = 0

TECH_SAFE_START_IDX = 20

# スプレッド（円）
SPREAD = 0.003
LEVERAGE = 25.0
MARGIN_RATIO = 1.0 / LEVERAGE
# 初期残高（円
INIT_BALANCE = 100000.0
PRICE_TO_PIPS = 100
PIPS_TO_PRICE = 0.01

FUTURE_LENGTH = 4


REWERD_MAX_PIPS = 100
REWARD_MAX_DIFF_PIPS = 10

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
        return price2pips(self.broker.now_price - self.open_price) * (-1 if self.is_short else 1)

    def close(self):
        self.close_time = self.broker.now_time
        self.close_price = self.broker.now_price
        return self.pl


class Broker():
    def __init__(self, leverage, balance, df, gasf, end_idx):
        # 現ステップでの時刻
        self.now_time = dt.datetime(2000, 1, 1)
        # 現ステップでのClose値
        self.now_price = 0
        self.prev_price = 0
        self.prev_pl_pips = 0
        self.future_price = 0

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

        self.end_idx = end_idx

        self.setup_bbandVol()
        self.vola_std = np.std(self.volatility_arr)
        self.vola_mean = np.mean(self.volatility_arr)
        #self.volatility_arr = []
        #self.setup_volatility_arr(self.df.Close.values.tolist(), 60)

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

    @property
    def unreal_pips(self):
        pips = 0
        for pos in self.positions:
            pips += pos.pips
        return pips

    # 必要証拠金(円)
    @property
    def margin(self):
        return self.position_size * MARGIN_RATIO * self.now_price

    @property
    def is_risky(self):
        return self.get_volatility(0) > self.vola_mean + self.vola_std*2

    def setup_bbandVol(self):
        # return self.volatility_arr[self.iter + delta]
        bband_diff = self.df['BBAND_U2'] - self.df['BBAND_L2']
        self.volatility_arr = bband_diff.clip(0.1, 10)

    def get_volatility(self, delta):
        return self.volatility_arr.iat[self.iter + delta]

    '''
    def get_volatility(self, delta):
        return self.volatility_arr[self.iter + delta]
    '''

    def get_gasf_data(self):
        return self.gasf[self.iter]

    # 内部状態を更新する（各ステップの最初に必ず呼び出す)
    # return: is_last
    def update(self):
        self.prev_pl_pips = self.unreal_pips
        self.prev_price = self.now_price

        self.iter += 1
        self.now_price = self.df.Close.iloc[self.iter]
        self.now_time = self.df.Datetime.iloc[self.iter]
        self.future_price = np.mean(self.df.Close.values[self.iter:self.iter +
                                                         FUTURE_LENGTH])
        return self.iter+FUTURE_LENGTH >= self.end_idx-1 or self.balance <= 0

    # ロスカットが必要かどうか
    def need_loss_cut(self):
        return self.margin > self.max_allowable_loss

    # ポジションを作成
    def open_position(self, size):
        spread = np.sign(size) * SPREAD
        pos = Position(self.now_price + spread, size, self)
        self.positions.append(pos)
        self.position_size += size

    def buy(self):
        self.open_position(self.trade_size)

    def sell(self):
        self.open_position(-self.trade_size)

    # ポジションを確定する
    def close(self, close_long=True, close_short=True):
        pl = 0
        pips = 0
        for pos in self.positions:
            # Long or Shortの条件に合致するなら
            if (pos.is_long and close_long) or (pos.is_short and close_short):
                # ポジションを確定して損益を計上
                pl += pos.close()
                pips += pos.pips
                self.position_size -= pos.size
                # 取引履歴に追加
                if pos.is_long:
                    self.long_hists.append(pos)
                elif pos.is_short:
                    self.short_hists.append(pos)
                # 保持ポジションリストから削除
                self.positions.remove(pos)
        self.balance += pl
        return pips


class FxEnv_GASF(gym.Env):
    TEST_MODE = 'test'
    TRAIN_MODE = 'train'

    def __init__(self, df, gasf, mode: str, trade_duration):
        self.df = df
        self.gasf = gasf
        self.mode = mode
        self.action_hist = [0, 0, 0]
        self.broker = None

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            # gasf+pos+pipsで3, ch, h, w
            shape=(2, self.gasf.shape[1], self.gasf.shape[2], self.gasf.shape[3], ))
        self.reward_range = (-1.0, 1.0)
        self.trade_duration = trade_duration

    def reset(self):
        self.action_hist = [0, 0, 0]

        if self.trade_duration:
            self.start_idx = random.randint(
                TECH_SAFE_START_IDX, len(self.df) - self.trade_duration)
            self.end_idx = self.start_idx + self.trade_duration
        else:
            self.start_idx = TECH_SAFE_START_IDX
            self.end_idx = len(self.df)

        self.broker = Broker(MARGIN_RATIO, INIT_BALANCE,
                             self.df, self.gasf, self.end_idx)
        for _ in range(self.start_idx):
            self.broker.update()
        return self.observe()

    def step(self, action):
        pips = 0

        # need_loss_cut = self.broker.need_loss_cut()
        if action == CLOSE:
            if len(self.broker.positions) > 0:
                pips = self.broker.close()
        elif action == BUY and not self.broker.has_long:
            pips = self.broker.close()
            self.broker.buy()
        elif action == SELL and not self.broker.has_short:
            pips = self.broker.close()
            self.broker.sell()
        self.action_hist.append(action2int(action))

        # 時間を経過させる
        done = self.broker.update()
        # 報酬を計算
        reward = self.calc_future_reward(action, pips)

        return self.observe(), reward, done, {'pips': pips}
        # return self.observe(), pips, done, {}

    def render(self):
        if self.mode == self.TEST_MODE:
            print('{:>3.1f}% ({}/{})  balance:{:>5.1f}  position:{:>3.1f}                      '.format(
                (self.broker.iter-self.start_idx) /
                (self.end_idx-self.start_idx) * 100, self.broker.iter-self.start_idx, self.end_idx-self.start_idx, self.broker.balance, self.broker.position_size), end='\r')

        elif self.mode == self.TRAIN_MODE:
            print('{:>3.1f}% ({}/{})  balance:{:>5.1f}  position:{:>3.1f}                      '.format(
                (self.broker.iter-self.start_idx) /
                (self.end_idx-self.start_idx) * 100, self.broker.iter-self.start_idx, self.end_idx-self.start_idx, self.broker.balance, self.broker.position_size), end='\r')

    def observe(self):
        gasf = self.broker.get_gasf_data()
        # shape = (1, gasf.shape[1], gasf.shape[2])
        pos = 0.0
        if self.broker.has_long:
            pos = 1.0
        elif self.broker.has_short:
            pos = -1.0
        # pos_arry = np.full(shape, pos)
        # pips_arry = np.full(shape, self.broker.unreal_pips/50)
        # np.concatenate([gasf, pos_arry, pips_arry], axis=0)
        linears = np.array([pos, 0], dtype=np.float32)
        obs = [gasf, linears]
        return obs

    # 利益計算（https://qiita.com/ryo_grid/items/1552d70eb2a8c15f6fd2 参照）
    def calc_rewerd(self):
        # 取引量（１でいいらしい
        mu = 1
        # 取引コスト
        bp = 0.0015
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

        diff = A1*rt*5/sigma1
        cost = bp*abs(5/sigma1 * A1 - 5/sigma2 * A2)

        return mu * (diff - cost)

    def calc_rewerd2(self, act, realized_pips):
        reward = 0
        # 取引確定時の利益を評価
        if realized_pips != 0:
            # BUY-SELL or SELL-BUY or CLOSE
            reward = np.sign(realized_pips) * \
                min(abs(realized_pips / REWERD_MAX_PIPS), 1)
        # CLOSE-CLOSEの取引なし。高ボラティリティでの取引はリスクが高い → 取引しなかったら高評価
        elif self.action_hist[-2] == CLOSE and self.action_hist[-1] == CLOSE:
            if self.broker.is_risky:
                reward = 1/REWERD_MAX_PIPS
        # BUY-BUY or SELL-SELL or CLOSE-BUY or CLOSE-SELL
        # 含み損益の増減を評価
        else:
            ret = self.broker.unreal_pips - self.broker.prev_pl_pips
            reward = np.sign(ret) / REWERD_MAX_PIPS
            # リスクのある取引を継続したら報酬1割引き
            if self.broker.is_risky:
                reward -= abs(reward) * 0.1
        return reward

    def calc_future_reward(self, act, pips):
        reward = 0
        # 損益が確定した場合
        if pips != 0:
            reward = np.sign(pips) * min(abs(pips / REWERD_MAX_PIPS), 1)
            # 切った場合
        elif act == 0:
            # ボラティリティが2σより高い場合（高リスク
            if self.broker.is_risky:
                # 高評価
                reward = min(abs(self.broker.future_price -
                                 self.broker.now_price) / REWERD_MAX_PIPS, 1)
        # 注文を作成・継続した場合
        else:
            # 将来の価格との差分を報酬に
            ret = (self.broker.future_price - self.broker.now_price)
            reward = np.sign(ret) * min(abs(ret / REWARD_MAX_DIFF_PIPS), 1)
            if act == SELL:
                reward *= - 1
        return reward

    def close(self):
        if mode == self.TEST_MODE:
            # グラフのプロット？
            pass
        elif mode == self.TRAIN_MODE:
            pass
