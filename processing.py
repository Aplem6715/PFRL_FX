
'''
参考（てかコピペ）
https://github.com/pecu/FinancialVision/tree/master/Encoding%20candlesticks%20as%20images%20for%20patterns%20classification%20using%20convolutional%20neural%20networks
'''

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


def ts2gasf(ts, max_v, min_v):
    '''
    Args:
        ts (numpy): (N, )
        max_v (int): max value for normalization
        min_v (int): min value for normalization
    Returns:
        gaf_m (numpy): (N, N)
    '''
    # Normalization : 0 ~ 1
    if max_v == min_v:
        gaf_m = np.zeros((len(ts), len(ts)))
    else:
        ts_nor = np.array((ts-min_v) / (max_v-min_v))
        # Arccos
        ts_nor_arc = np.arccos(ts_nor)
        # GAF
        gaf_m = np.zeros((len(ts_nor), len(ts_nor)))
        for r in range(len(ts_nor)):
            for c in range(len(ts_nor)):
                gaf_m[r, c] = np.cos(ts_nor_arc[r] + ts_nor_arc[c])
    return gaf_m


def get_gasf(arr):
    '''Convert time-series to gasf    
    Args:
        arr (numpy): (N, ts_n, 4)
    Returns:
        gasf (numpy): (N, ts_n, ts_n, 4)
    Todos:
        add normalization together version
    '''
    arr = arr.copy()
    gasf = np.zeros((arr.shape[0], arr.shape[1], arr.shape[1], arr.shape[2]))
    for i in range(arr.shape[0]):
        for c in range(arr.shape[2]):
            each_channel = arr[i, :, c]
            c_max = np.amax(each_channel)
            c_min = np.amin(each_channel)
            each_gasf = ts2gasf(each_channel, max_v=c_max, min_v=c_min)
            gasf[i, :, :, c] = each_gasf
    return gasf


def ohlc2culr(ohlc):
    '''
    Args:
        ohlc (numpy): (N, ts_n, 4)
    Returns:
        culr (numpy): (N, ts_n, 4)
    '''
    culr = np.zeros((ohlc.shape[0], ohlc.shape[1], ohlc.shape[2]))
    culr[:, :, 0] = ohlc[:, :, -1]
    culr[:, :, 1] = ohlc[:, :, 1] - np.maximum(ohlc[:, :, 0], ohlc[:, :, -1])
    culr[:, :, 2] = np.minimum(ohlc[:, :, 0], ohlc[:, :, -1]) - ohlc[:, :, 2]
    culr[:, :, 3] = ohlc[:, :, -1] - ohlc[:, :, 0]
    return culr


def create_ts_with_window(ts2d, window_size):
    '''
    Args:
        ts2d (numpy): (N, 4)
        ohlc とか culr
    Returns:
        ts_data (numpy): (N, window_size, 4)
    '''
    ts_data = np.zeros((ts2d.shape[0], window_size, ts2d.shape[1]))
    for i in range(window_size-1, ts2d.shape[0]):
        ts_data[i, :, :] = ts2d[i - window_size + 1: i + 1, :]
    return ts_data


def get_ohlc_gasf(ohlc_df):
    ts_data = create_ts_with_window(df.values, 32)
    gasf = get_gasf(ts_data)
    return gasf


def get_culr_gasf(ohlc_df):
    ts_data = create_ts_with_window(df.values, 32)
    ts_data = ohlc2culr(ts_data)
    gasf = get_gasf(ts_data)
    return gasf


def get_ohlc_culr_gasf(ohlc_df):
    ohlc_data = create_ts_with_window(ohlc_df.values, 32)
    culr_data = ohlc2culr(ohlc_data)
    ohlc_gasf = get_gasf(ohlc_data)
    culr_gasf = get_gasf(culr_data)
    con = np.concatenate([ohlc_gasf, culr_gasf], axis=3)
    return con


if __name__ == '__main__':
    df = pd.read_csv('M30_201001-201912_Tech7.csv', parse_dates=[0])
    df = df[((df['Datetime'] >= dt.datetime(2017, 12, 1))
             & (df['Datetime'] < dt.datetime(2018, 1, 1)))]
    df = df.loc[:, 'Open': 'Close']
    ts_data = get_ohlc_culr_gasf(df)
    plt.imshow(ts_data[32, :, :, 0], cmap='gray', vmin=-1,
               vmax=1, interpolation='none')
    plt.show()
    print(ts_data.shape)

'''
    plt.imshow(gasf[32, :, :, 0], cmap='gray', vmin=-1,
               vmax=1, interpolation='none')
    plt.show()
    plt.imshow(gasf[32, :, :, 1], cmap='gray', vmin=-1,
               vmax=1, interpolation='none')
    plt.show()
    plt.imshow(gasf[32, :, :, 2], cmap='gray', vmin=-1,
               vmax=1, interpolation='none')
    plt.show()
    plt.imshow(gasf[32, :, :, 3], cmap='gray', vmin=-1,
               vmax=1, interpolation='none')
    plt.show()
'''
