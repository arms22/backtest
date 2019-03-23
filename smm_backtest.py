# -*- coding: utf-8 -*-
from backtest.core import Backtest, BacktestIteration
from backtest.indicator import *
import pandas as pd
from hyperopt import hp

def simple_market_make_backtest(ohlcv, period, margin):

    def smm_logic1(O, H, L, C, n, strategy):
        maxsize = 0.1
        buysize = sellsize = 0.1
        spr = ohlcv.stdev[n]*2.5
        mid = (C+H+L)/3
        buy = mid - spr/2
        sell = mid + spr/2
        if strategy.position_size < maxsize:
            strategy.order('L', 'buy', qty=buysize, limit=buy)
        else:
            strategy.cancel('L')
        if strategy.position_size > -maxsize:
            strategy.order('S', 'sell', qty=sellsize, limit=sell)
        else:
            strategy.cancel('S')

    std = stdev(ohlcv.close, period).values
    z = zscore(ohlcv.volume_imbalance, 300).values

    def smm_logic2(O, H, L, C, n, strategy):
        spr = max(std[n], margin)
        # spr = margin
        pairs = [(0.02, spr*1.0, 3), (0.02, spr*0.5, 2), (0.02, spr*0.25, 1), (0.02, spr*0.125, 0)]
        maxsize = sum(p[0] for p in pairs) - 0.01
        # maxsize = 0.2
        buymax = sellmax = strategy.position_size
        mid = (C+H+L)/3
        mid = mid + z[n]*9
        for pair in pairs:
            suffix = str(pair[2])
            if buymax+pair[0] <= maxsize:
                buymax += pair[0]
                strategy.order('L'+suffix, 'buy', qty=pair[0], limit=mid-pair[1])
            else:
                strategy.cancel('L'+suffix)
            if sellmax-pair[0] >= -maxsize:
                sellmax -= pair[0]
                strategy.order('S'+suffix, 'sell', qty=pair[0], limit=mid+pair[1])
            else:
                strategy.cancel('S'+suffix)

    yourlogic = smm_logic1
    # yourlogic = smm_logic2

    return Backtest(**locals())

if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bffx_2019-03-21_5s.csv', index_col="time", parse_dates=True)

    default_parameters = {
        'ohlcv': ohlcv,
        'period':12*15,
        'margin':400,
    }

    hyperopt_parameters = {
        'period': hp.quniform('period',  2, 1000, 1),
        # 'margin': hp.quniform('margin', 10, 2000, 1),
    }

    best, report = BacktestIteration(simple_market_make_backtest, default_parameters, hyperopt_parameters, 0)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
