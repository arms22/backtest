# -*- coding: utf-8 -*-
from backtest.core import Backtest, BacktestIteration
from backtest.indicator import *
import pandas as pd
from hyperopt import hp

def market_make_backtest(ohlcv, margin):

    limit_buy_entry = ohlcv.close - margin
    limit_buy_exit = ohlcv.close + margin
    limit_sell_entry = ohlcv.close + margin
    limit_sell_exit = ohlcv.close - margin

    return Backtest(**locals())

if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bffx_2019-03-21_5s.csv', index_col="time", parse_dates=True)

    default_parameters = {
        'ohlcv': ohlcv,
        'margin': 47,
    }

    hyperopt_parameters = {
        'margin': hp.uniform('margin', 10, 10000),
    }

    def maximize(r):
        # return ((r.All.WinRatio * r.All.WinPct) + ((1 - r.All.WinRatio) * r.All.LossPct)) * r.All.Trades
        # return r.All.WinPct * r.All.WinRatio * r.All.WinTrades
        return r.All.Profit
        # return r.All.ProfitFactor

    best, report = BacktestIteration(market_make_backtest, default_parameters, hyperopt_parameters, 0, maximize=maximize)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
