# -*- coding: utf-8 -*-
from backtest.core import Backtest, BacktestIteration
from backtest.indicator import *
import pandas as pd
from hyperopt import hp

def fraction_backtest(ohlcv):

    std = stdev(ohlcv.close,50).values
    def yourlogic(O, H, L, C, n, strategy):
        maxsize = 0.1
        mid = (C+H+L)/3
        # mid = C
        lot = 0.01
        rng = 51
        spr = std[n]*0.666
        buy = ((mid-spr)//rng)*rng
        sell = ((mid+spr)//rng+1)*rng
        if n%4==0:
            if strategy.position_size<maxsize:
                strategy.order(buy, 'buy', qty=lot, limit=buy)
            if strategy.position_size>-maxsize:
                strategy.order(sell, 'sell', qty=lot, limit=sell)
        profit = rng
        loss = -rng*0.5
        for p in strategy.positions:
            side, price, size = p
            if size>=0.01:
                if side>0:
                    pnl = mid-price
                    if pnl<=loss or pnl>=profit:
                        strategy.order(-price, 'sell', qty=size)
                else:
                    pnl = price-mid
                    if pnl<=loss or pnl>=profit:
                        strategy.order(-price, 'buy', qty=size)

    ohlcv['delay'] = 3

    return Backtest(**locals())

if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bffx_2019-03-25_500ms.csv', index_col="exec_date", parse_dates=True)

    # デフォルトパラメータ
    default_parameters = {
        'ohlcv': ohlcv,
    }

    # 探索パラメータ
    hyperopt_parameters = {
    }

    best, report = BacktestIteration(fraction_backtest, default_parameters, hyperopt_parameters, 0)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
