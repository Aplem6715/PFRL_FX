import gym
import gym.spaces
import pandas as pd
import numpy as np
from typing import List

# df.max().values[1:].max()
TECH_FILE_VALUE_MAX = 130.0
# df.min().values[1:].min()
TECH_FILE_VALUE_MIN = -6.0


class Position():
    def __init__(self, is_long: bool, lots: int, open_pri: float):
        self.open_price = open_pri
        self.lots = lots
        self.is_long = is_long

    def get_pl(self, close_pri: float):
        if self.is_long:
            return (close_pri - self.open_price)*self.lots
        else:
            return (self.open_price - close_pri)*self.lots


class Account():
    def __init__(self, balance: float):
        self.balance = balance  # type: float
        self.positions = []  # type: List[Position]

    @property
    def has_position(self):
        return len(self.positions) != 0


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
        self.tech_file_path = './fx_env/fx_env/envs/M30_201001-201912_Tech9.csv'
        df = pd.read_csv(self.tech_file_path, parse_dates=[0])
        self.data = df.values[1:]  # np.ndarray
        self.data_iter = 0

        # 初期値の定義
        self.init_balance = 10000

        # FX取引に関する定数
        self.spread = 0.8
        self.leverage = 25

        # 取引に関する変数
        self.now_price = None

        # 環境の設定
        self.action = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=TECH_FILE_VALUE_MIN, high=TECH_FILE_VALUE_MAX,
            shape=(self.window_size, self.data.shape[1]))
        return

    def _reset(self):
        self.account = Account(self.init_balance)
        self.data_iter = self.window_size
        return self._observe()

    def _step(self):
        pass

    def _observe(self):
        return self.data[self.data_iter-self.window_size:self.data_iter]

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

    def buy(self, lots: int):
        # shortポジションをクローズ
        pl = self.close(is_long=False)
        # 買いポジションを追加
        self.account.positions.append(
            Position(is_long=True, lots=lots, open_pri=self.now_price))
        return pl

    def sell(self, lots):
        # longポジションをクローズ
        pl = self.close(is_long=True)
        # 売りポジションを追加
        self.account.positions.append(
            Position(is_long=False, lots=lots, open_pri=self.now_price))
        return pl


env = FxEnv()
print(env.observation_space.shape)
