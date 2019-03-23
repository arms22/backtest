# -*- coding: utf-8 -*-
from backtest.core import Backtest, BacktestIteration
from backtest.indicator import *
import pandas as pd
from hyperopt import hp

def channel_breakout_backtest(ohlcv, breakout_in, breakout_out):

    stop_buy_entry = highest(ohlcv.high, breakout_in) + 0.5
    stop_buy_exit = lowest(ohlcv.low, breakout_out) - 0.5
    stop_sell_entry = lowest(ohlcv.low, breakout_in) - 0.5
    stop_sell_exit = highest(ohlcv.high, breakout_out) + 0.5

    return Backtest(**locals())

if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bitmex_2019_1h.csv', index_col='timestamp', parse_dates=True)

    default_parameters = {
        'ohlcv':ohlcv,
        'breakout_in':18,
        'breakout_out':9,
    }

    hyperopt_parameters = {
        'breakout_in': hp.quniform('breakout_in', 1, 100, 1),
        'breakout_out': hp.quniform('breakout_out', 1, 100, 1),
    }

    best, report = BacktestIteration(channel_breakout_backtest, default_parameters, hyperopt_parameters, 0)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
