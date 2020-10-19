import numpy as np

LEVERAGE = 25.0
SPREAD = 0.003
MARGIN_RATIO = 1 / LEVERAGE


# 必要証拠金を計算
def calc_margin(position_unit: float, now_price: float):
    return position_unit * MARGIN_RATIO * now_price


# 損益を計算
def calc_pl(open_price, now_price, units):
    return (now_price - open_price) * LEVERAGE * units - SPREAD * LEVERAGE * np.sign(units)
