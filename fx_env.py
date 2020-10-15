import gym
import gym.spaces
import pandas as pd
import numpy as np
import datetime as dt
from sklearn import preprocessing
from typing import List

# df.max().values[1:].max()
TECH_FILE_VALUE_MAX = 130.0
# df.min().values[1:].min()
TECH_FILE_VALUE_MIN = -6.0

# FX取引に関する定数
spread = 0.008
trade_lots = 10


class Position():
    def __init__(self, is_long: bool, lots: int, open_pri: float):
        self.open_price = open_pri
        self.lots = lots
        self.is_long = is_long

    def get_pl(self, close_pri: float):
        if self.is_long:
            return (close_pri - self.open_price + spread)*self.lots
        else:
            return (self.open_price - close_pri - spread)*self.lots


class Account():
    def __init__(self, balance: float):
        self.balance = balance  # type: float
        self.positions = []  # type: List[Position]

    @property
    def has_position(self):
        return len(self.positions) != 0

    def take_pl(self, pl):
        self.balance += pl


class FxEnv(gym.Env):
    def __init__(self):
        # 定数の定義
        self.STAY = 0
        self.BUY = 1
        self.SELL = 2
        self.CLOSE = 3

        # 学習に関する設定
        self.window_size = 5  # 過去何本分のロウソクを見るか

        # OHLCデータ
        self.tech_file_path = 'M30_201001-201912_Tech9.csv'
        df = pd.read_csv(self.tech_file_path, parse_dates=[0])
        df = df[((df['Datetime'] >= dt.datetime(2017, 6, 1))
                 & (df['Datetime'] < dt.datetime(2018, 1, 1)))]
        padding = len(df) % self.window_size
        data = df.values[:-padding, 1:]
        # データの正規化
        scaler = preprocessing.MinMaxScaler()
        self.data = scaler.fit_transform(data)

        self.data_iter = 0

        # 初期値の定義
        self.init_balance = 10000

        # 環境の設定
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=TECH_FILE_VALUE_MIN, high=TECH_FILE_VALUE_MAX,
            shape=(self.window_size, self.data.shape[1]))
        return

    @property
    def now_price(self):
        return self.data[self.data_iter][3]

    @property
    def done(self):
        # データの終端 or 残高が0で終了
        return (self.data_iter + 1 >= len(self.data)) or self.account.balance <= trade_lots

    def reset(self):
        self.account = Account(self.init_balance)
        self.data_iter = self.window_size - 1

        return self._observe()

    def step(self, action):
        reward = 0
        if action == self.STAY:
            pass
        elif action == self.BUY:
            reward = self.buy()
        elif action == self.SELL:
            reward = self.sell()
        elif action == self.CLOSE:
            reward = self.close()
        self.account.take_pl(reward)
        self.data_iter += 1

        if self.account.balance < 0:
            reward = -100

        return self._observe(), reward, self.done, self._info()

    def render(self):
        print('{:.1f}% ({}/{})  balance:{:.1f}'.format(self.data_iter/len(self.data)
                                                       * 100, self.data_iter, len(self.data), self.account.balance), end='\r')
        return

    def _observe(self):
        return self.data[self.data_iter - self.window_size + 1: self.data_iter + 1].ravel()

    def _info(self):
        return {'balance': self.account.balance}

    # すべてのポジションを閉じて損益を返す
    def close(self, is_long=None):
        pl = 0
        for pos in self.account.positions:
            if (is_long is not None):
                # long/shortが指定されている場合には指定と合致するもののみクローズ
                if is_long == pos.is_long:
                    pl += pos.get_pl(self.now_price)
                    self.account.positions.remove(pos)
            else:
                # long/shortの指定がないときはすべてクローズ
                pl += pos.get_pl(self.now_price)
                self.account.positions.remove(pos)
        return pl

    # 買い注文を出し，損益を返す
    def buy(self):
        # shortポジションをクローズ
        pl = self.close(is_long=False)
        # 買いポジションを追加
        self.account.positions.append(
            Position(is_long=True, lots=trade_lots, open_pri=self.now_price))
        return pl

    def sell(self):
        # longポジションをクローズ
        pl = self.close(is_long=True)
        # 売りポジションを追加
        self.account.positions.append(
            Position(is_long=False, lots=trade_lots, open_pri=self.now_price))
        return pl
