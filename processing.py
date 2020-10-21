
'''
参考（てかコピペ）
https://github.com/pecu/FinancialVision/tree/master/Encoding%20candlesticks%20as%20images%20for%20patterns%20classification%20using%20convolutional%20neural%20networks
'''

import numpy as np


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