# -*- coding: utf-8 -*-
from backtest.core import Backtest, BacktestIteration
from backtest.indicator import *
import pandas as pd
from hyperopt import hp

def simple_market_make_backtest(ohlcv, period, margin):

    std = stdev3(ohlcv,period).values
    z = zscore(ohlcv.volume_imbalance, period).values*9
    delay = ohlcv.delay.rolling(3).median().values
    coffeebreak = ohlcv.mente.values

    def yourlogic(O, H, L, C, n, strategy):
        if not coffeebreak[n]:
            spr = max(std[n], margin)
            pairs = [(0.05, spr*0.50, '2', 2), (0.05, spr*0.25, '1', 1)]
            maxsize = sum(p[0] for p in pairs)
            buymax = sellmax = strategy.position_size
            mid = C
            ofs = z[n]
            if delay[n]>2.5:
                if delay[n]>2.5:
                    if strategy.position_size>0:
                        strategy.order('Lc', 'sell', qty=0.01)
                    elif strategy.position_size<0:
                        strategy.order('Sc', 'buy', qty=0.01)
                for _,_,suffix,_ in pairs:
                    strategy.cancel('L'+suffix)
                    strategy.cancel('S'+suffix)
            else:
                for size, width, suffix, period in pairs:
                    buyid = 'L'+suffix
                    sellid = 'S'+suffix
                    buysize = min(maxsize-buymax,size)
                    if buymax+buysize <= maxsize and buysize>0:
                        if ((n%period)==0 or buyid not in strategy.open_orders):
                            strategy.order(buyid, 'buy', qty=buysize, limit=mid-width+ofs)
                        buymax += buysize
                    else:
                        strategy.cancel(buyid)
                    sellsize = min(maxsize+sellmax,size)
                    if sellmax-sellsize >= -maxsize and sellsize>0:
                        if (n%period)==0 or sellid not in strategy.open_orders:
                            strategy.order(sellid, 'sell', qty=sellsize, limit=mid+width+ofs)
                        sellmax -= sellsize
                    else:
                        strategy.cancel(sellid)
        else:
            strategy.cancel_order_all()
            strategy.close_position()

    return Backtest(**locals())

if __name__ == '__main__':
    from datetime import timedelta

    # テストデータ読み込み
    ohlcv = pd.concat([
        pd.read_csv('csv/bffx_2019-04-12_5s.csv', index_col="time", parse_dates=True),
        ])
    boundary = ohlcv.index.to_series().diff()>timedelta(hours=1)
    ohlcv['delay'] = ohlcv['delay'].clip_lower(0)
    ohlcv['mente'] = boundary.shift(4)|boundary.shift(3)|boundary.shift(2)|boundary.shift(1)|boundary|boundary.shift(-1)|boundary.shift(-2)|boundary.shift(-3)|boundary.shift(-4)

    default_parameters = {
        'ohlcv': ohlcv,
        'period':12*20,
        'margin':1000,
    }

    hyperopt_parameters = {
        'period': hp.quniform('period',  2, 300, 2),
        'margin': hp.quniform('margin', 10, 2000, 10),
    }

    best, report = BacktestIteration(simple_market_make_backtest, default_parameters, hyperopt_parameters, 0)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
