import numpy as np
import math

LEVERAGE = 25.0
SPREAD = 0.003
MARGIN_RATIO = 1 / LEVERAGE


# 必要証拠金を計算
def calc_margin(position_unit: float, now_price: float):
    return position_unit * MARGIN_RATIO * now_price


# 損益を計算
def calc_pl(open_price, now_price, units):
    return (now_price - open_price) * LEVERAGE * units - SPREAD * LEVERAGE * np.sign(units)


def calculate_volatility(partial_price_arr, window_size):
    alpha = 2 / float(window_size + 1)
    ema_arr = []
    emvar_arr = []
    delta = 0
    ema_arr.append(partial_price_arr[0])
    emvar_arr.append(delta)
    for idx in range(1, window_size):
        delta = partial_price_arr[idx] - ema_arr[idx - 1]
        ema_arr.append(ema_arr[idx - 1] + alpha * delta)
        emvar_arr.append(
            (1 - alpha) * (emvar_arr[idx - 1] + alpha * delta * delta))

    emsd = math.sqrt(emvar_arr[-1])
    return emsd
