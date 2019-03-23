# -*- coding: utf-8 -*-
from backtest.core import Backtest, BacktestIteration
from backtest.indicator import *
import pandas as pd
from hyperopt import hp
import sys
from functools import lru_cache

def macd_cross_backtest(ohlcv, fastlen, slowlen, siglen):

    @lru_cache(maxsize=None)
    def cached_sma(period):
        return ohlcv.close.rolling(int(period)).mean()

    @lru_cache(maxsize=None)
    def cached_macd(fastlen, slowlen, siglen):
        macd = cached_sma(fastlen) - cached_sma(slowlen)
        signal = macd.rolling(int(siglen)).mean()
        return (macd, signal, macd-signal)

    @lru_cache(maxsize=None)
    def cached_atr(period, multi):
        return atr(ohlcv.close, ohlcv.high, ohlcv.low, period) * multi

    # インジケーター作成
    vmacd, vsig, vhist = cached_macd(fastlen, slowlen, siglen)

    # エントリー／イグジット
    buy_entry = crossover(vmacd, vsig)
    sell_entry = crossunder(vmacd, vsig)
    buy_exit = sell_entry.copy()
    sell_exit = buy_entry.copy()

    ignore = int(max([fastlen, slowlen]))
    buy_entry[:ignore] = False
    buy_exit[:ignore] = False
    sell_entry[:ignore] = False
    sell_exit[:ignore] = False

    # ゼロラインフィルタ
    if 0:
        buy_entry[vsig>0] = False
        sell_entry[vsig<0] = False

    # ハイローフィルタ
    if 0:
        macd_high = highest(vmacd, smaslowlen)
        macd_low = lowest(vmacd, smafastlen)
        macd_middle = (macd_high + macd_low) / 2
        buy_entry[vmacd > macd_middle] = False
        sell_entry[vmacd < macd_middle] = False
        buy_exit[vmacd < macd_middle] = False
        sell_exit[vmacd > macd_middle] = False

    # ATRによるSTOP注文
    if 0:
        range = cached_atr(5, 1.6)
        downtrendsfilter = ohlcv.close < cached_sma(120)
        stop_buy_entry = ohlcv.high + range
        stop_sell_exit = ohlcv.high + range
        stop_buy_entry[downtrendsfilter] = 0
        stop_sell_exit[downtrendsfilter] = 0
    if 0:
        range = cached_atr(5, 1.6)
        uptrendsfilter = ohlcv.close > cached_sma(120)
        stop_sell_entry = ohlcv.low - range
        stop_buy_exit = ohlcv.low - range
        stop_sell_entry[uptrendsfilter] = 0
        stop_buy_exit[uptrendsfilter] = 0

    return Backtest(**locals())

if __name__ == '__main__':

    import argparse
    import datetime
    import dateutil.parser

    def store_datetime(str):
        return dateutil.parser.parse(str)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('csv', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument("--start", type=store_datetime)
    parser.add_argument("--end", type=store_datetime)
    parser.add_argument("--max_evals", dest='max_evals', type=int, default=0)
    args = parser.parse_args()

    ohlcv = pd.read_csv(args.csv, index_col='timestamp', parse_dates=True)
    if args.start is not None and args.end is not None:
        ohlcv = ohlcv[args.start:args.end]
    elif args.start is None:
        ohlcv = ohlcv[:args.end]
    elif args.end is None:
        ohlcv = ohlcv[args.start:]

    default_parameters = {
        'ohlcv': ohlcv,
        'fastlen':19,
        'slowlen':27,
        'siglen':13,
        # 'fastlen':17,
        # 'slowlen':30,
        # 'siglen':18,
        # 'fastlen':16,
        # 'slowlen':26,
        # 'siglen':9,
    }

    hyperopt_parameters = {
        'fastlen': hp.quniform('fastlen', 5, 50, 1),
        'slowlen': hp.quniform('slowlen', 5, 50, 1),
        'siglen': hp.quniform('siglen', 1, 50, 1),
    }

    best, report = BacktestIteration(macd_cross_backtest, default_parameters, hyperopt_parameters, args.max_evals)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
