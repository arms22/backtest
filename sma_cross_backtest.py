# -*- coding: utf-8 -*-
from backtest.core import Backtest, BacktestIteration
from backtest.indicator import *
import pandas as pd
from hyperopt import hp

def sma_cross_backtest(ohlcv, fastlen, slowlen):

    # インジケーター作成
    vfast = sma(ohlcv.close, fastlen)
    vslow = sma(ohlcv.close, slowlen)

    # エントリー／イグジット
    buy_entry = crossover(vfast, vslow)
    buy_exit = crossunder(vfast, vslow)
    sell_entry = crossunder(vfast, vslow)
    sell_exit = crossover(vfast, vslow)

    return Backtest(**locals())

if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bitmex_2019_1h.csv', index_col='timestamp', parse_dates=True)

    default_parameters = {
        'ohlcv': ohlcv,
        'fastlen':38,
        'slowlen':29,
    }

    hyperopt_parameters = {
        'fastlen': hp.quniform('fastlen', 1, 100, 1),
        'slowlen': hp.quniform('slowlen', 1, 200, 1),
    }

    best, report = BacktestIteration(sma_cross_backtest, default_parameters, hyperopt_parameters, 0, maximize=lambda r:r.All.Profit)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
