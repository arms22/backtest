# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from numba import jit, b1, f8, i8, void
from .utils import dotdict
from hyperopt import hp, tpe, Trials, fmin, rand, anneal
from collections import deque

# PythonでFXシストレのバックテスト(1)
# https://qiita.com/toyolab/items/e8292d2f051a88517cb2 より

@jit(f8(f8,f8,f8,f8,f8,f8,f8),nopython=True)
def buy_order(market, limit, stop, O, H, L, C):
    exec_price = 0
    # STOP注文
    if stop > 0 and H >= stop:
        if stop >= O:
            exec_price = stop
        else:
            exec_price = O
    # 指値注文
    elif limit > 0 and L <= limit:
        exec_price = limit
    # 成行注文
    elif market:
        exec_price = O
    # 注文執行
    return exec_price

@jit(f8(f8,f8,f8,f8,f8,f8,f8),nopython=True)
def buy_close(profit, stop, exec_price, O, H, L, C):
    close_price = 0
    if stop > 0:
        # 損切判定
        stop_price = exec_price - stop
        if L <= stop_price:
            close_price = stop_price
    if profit > 0:
        # 利確判定
        profit_price = exec_price + profit
        if H >= profit_price:
            close_price = profit_price
    return close_price

@jit(f8(f8,f8,f8,f8,f8,f8,f8),nopython=True)
def sell_order(market, limit, stop, O, H, L, C):
    exec_price = 0
    # STOP注文
    if stop > 0 and L <= stop:
        if stop <= O:
            exec_price = stop
        else:
            exec_price = O
    # 指値注文
    elif limit > 0 and H >= limit:
        exec_price = limit
    # 成行注文
    elif market:
        exec_price = O
    # 注文執行
    return exec_price

@jit(f8(f8,f8,f8,f8,f8,f8,f8),nopython=True)
def sell_close(profit, stop, exec_price, O, H, L, C):
    close_price = 0
    if stop > 0:
        # 損切判定
        stop_price = exec_price + stop
        if H >= stop_price:
            close_price = stop_price
    if profit > 0:
        # 利確判定
        profit_price = exec_price - profit
        if L <= profit_price:
            close_price = profit_price
    return close_price

@jit(f8(f8,f8,f8,f8),nopython=True)
def calclots(capital, price, percent, lot):
    if percent > 0:
        if capital > 0:
            return ((capital * percent) / price)
        else:
            return 0
    else:
        return lot

@jit(void(f8[:],f8[:],f8[:],f8[:],f8[:],i8[:],i8,
    b1[:],b1[:],b1[:],b1[:],
    f8[:],f8[:],f8[:],f8[:],
    f8[:],f8[:],f8[:],f8[:],
    f8[:],f8[:],f8,f8,
    f8,f8,f8,f8,f8,f8,f8,i8,i8,i8,f8,i8,
    f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:]), nopython=True)
def BacktestCore(Open, High, Low, Close, Volume, Trades, N,
    buy_entry, sell_entry, buy_exit, sell_exit,
    stop_buy_entry, stop_sell_entry, stop_buy_exit, stop_sell_exit,
    limit_buy_entry, limit_sell_entry, limit_buy_exit, limit_sell_exit,
    buy_size, sell_size, max_buy_size, max_sell_size,
    spread, take_profit, stop_loss, trailing_stop, slippage, percent, capital,
    trades_per_n, delay_n, order_restrict_n, max_drawdown, wait_n_for_mdd,
    LongTrade, LongPL, LongPct, ShortTrade, ShortPL, ShortPct, PositionSize):

    buyExecPrice = sellExecPrice = 0.0 # 売買価格
    buyMarketEntry = buyMarketExit = sellMarketEntry = sellMarketExit = 0
    buyStopEntry = buyStopExit = sellStopEntry = sellStopExit = 0
    buyLimitEntry = buyLimitExit = sellLimitEntry = sellLimitExit = 0
    buyExecLot = sellExecLot = 0
    buyEntryAcceptN = sellEntryAcceptN = buyExitAcceptN = sellExitAcceptN =0
    dd = max_profit = dd_wait = 0

    for n in range(delay_n, N):
        # O, H, L, C = Open[n], High[n], Low[n], Close[n]
        BuyNow = SellNow = False

        # ドローダウンが最大値を超えていたら一定時間取引停止
        EntryReject = dd_wait > 0
        if dd_wait > 0:
            dd_wait = dd_wait - 1

        # 約定数が規定値を超えていたら注文拒否
        OrderReject = (trades_per_n and Trades[n] > trades_per_n)

        # 買い注文処理
        if buyExecLot < max_buy_size:
            # 新規注文受付
            if not OrderReject and not EntryReject:
                if n>buyEntryAcceptN+order_restrict_n:
                    buyMarketEntry = buy_entry[n-delay_n]
                    buyLimitEntry = limit_buy_entry[n-delay_n]
                    buyStopEntry = stop_buy_entry[n-delay_n]
                    buyOpenSize = buy_size[n-delay_n]
                    if buyMarketEntry>0 or buyLimitEntry>0 or buyStopEntry>0:
                        buyEntryAcceptN = n
            OpenPrice = 0
            # 指値注文
            if buyLimitEntry > 0 and Low[n] <= buyLimitEntry:
                OpenPrice = buyLimitEntry
                buyLimitEntry = 0
            # STOP注文
            if buyStopEntry > 0 and High[n] >= buyStopEntry:
                if Open[n] <= buyStopEntry:
                    OpenPrice = buyStopEntry
                else:
                    OpenPrice = Open[n]
                buyStopEntry = 0
            # 成行注文
            if buyMarketEntry > 0:
                OpenPrice = Open[n]
                buyMarketEntry = 0
            # 注文執行
            if OpenPrice > 0 and Volume[n]>buyOpenSize:
                execPrice = OpenPrice + spread + slippage
                LongTrade[n] = execPrice #買いポジションオープン
                execLot =  calclots(capital, OpenPrice, percent, buyOpenSize)
                buyExecPrice = ((execPrice*execLot)+(buyExecPrice*buyExecLot))/(buyExecLot+execLot)
                buyExecLot = buyExecLot + execLot
                BuyNow = True

        # 買い手仕舞い
        if buyExecLot > 0 and not BuyNow:
            # 決済注文受付
            if not OrderReject:
                if n>buyExitAcceptN+order_restrict_n:
                    buyMarketExit = buy_exit[n-delay_n]
                    buyLimitExit = limit_buy_exit[n-delay_n]
                    buyStopExit = stop_buy_exit[n-delay_n]
                    buyCloseSize = buy_size[n-delay_n]
                    if buyMarketExit>0 or buyLimitExit>0 or buyStopExit>0:
                        buyExitAcceptN = n
            ClosePrice = 0
            # 指値注文
            if buyLimitExit > 0 and High[n] >= buyLimitExit:
                ClosePrice = buyLimitExit
                buyLimitExit = 0
            # STOP注文
            if buyStopExit > 0 and Low[n] <= buyStopExit:
                if Open[n] >= buyStopExit:
                    ClosePrice = buyStopExit
                else:
                    ClosePrice = Open[n]
                buyStopExit = 0
            # 成行注文
            if buyMarketExit > 0:
                ClosePrice = Open[n]
                buyMarketExit = 0
            # 注文執行
            if ClosePrice > 0 and Volume[n]>buyCloseSize:
                if buyExecLot > buyCloseSize:
                    buy_exit_lot = buyCloseSize
                    buy_exec_price = buyExecPrice
                    buyExecLot = buyExecLot - buy_exit_lot
                else:
                    buy_exit_lot = buyExecLot
                    buy_exec_price = buyExecPrice
                    buyExecPrice = buyExecLot = 0
                ClosePrice = ClosePrice - slippage
                LongTrade[n] = ClosePrice #買いポジションクローズ
                LongPL[n] = (ClosePrice - buy_exec_price) * buy_exit_lot #損益確定
                LongPct[n] = LongPL[n] / buy_exec_price

        # 売り注文処理
        if sellExecLot < max_sell_size:
            # 新規注文受付
            if not OrderReject and not EntryReject:
                if n>sellEntryAcceptN+order_restrict_n:
                    sellMarketEntry = sell_entry[n-delay_n]
                    sellLimitEntry = limit_sell_entry[n-delay_n]
                    sellStopEntry = stop_sell_entry[n-delay_n]
                    sellOpenSize = sell_size[n-delay_n]
                    if sellMarketEntry>0 or sellLimitEntry>0 or sellStopEntry>0:
                        sellEntryAcceptN = n
            OpenPrice = 0
            # 指値注文
            if sellLimitEntry > 0 and High[n] >= sellLimitEntry:
                OpenPrice = sellLimitEntry
                sellLimitEntry = 0
            # STOP注文
            if sellStopEntry > 0 and Low[n] <= sellStopEntry:
                if Open[n] >= sellStopEntry:
                    OpenPrice = sellStopEntry
                else:
                    OpenPrice = Open[n]
                sellStopEntry = 0
            # 成行注文
            if sellMarketEntry > 0:
                OpenPrice = Open[n]
                sellMarketEntry = 0
            # 注文執行
            if OpenPrice and Volume[n]>sellOpenSize:
                execPrice = OpenPrice - slippage
                ShortTrade[n] = execPrice #売りポジションオープン
                execLot = calclots(capital,OpenPrice,percent,sellOpenSize)
                sellExecPrice = ((execPrice*execLot)+(sellExecPrice*sellExecLot))/(sellExecLot+execLot)
                sellExecLot = sellExecLot + execLot
                SellNow = True

        # 売り手仕舞い
        if sellExecLot > 0 and not SellNow:
            # 決済注文受付
            if not OrderReject:
                if n>sellExitAcceptN+order_restrict_n:
                    sellMarketExit = sell_exit[n-delay_n]
                    sellLimitExit = limit_sell_exit[n-delay_n]
                    sellStopExit = stop_sell_exit[n-delay_n]
                    sellCloseSize = sell_size[n-delay_n]
                    if sellMarketExit>0 or sellLimitExit>0 or sellStopExit>0:
                        sellExitAcceptN = n
            ClosePrice = 0
            # 指値注文
            if sellLimitExit > 0 and Low[n] <= sellLimitExit:
                ClosePrice = sellLimitExit
                sellLimitExit = 0
            # STOP注文
            if sellStopExit > 0 and High[n] >= sellStopExit:
                if Open[n] <= sellStopExit:
                    ClosePrice = sellStopExit
                else:
                    ClosePrice = Open[n]
                sellStopExit = 0
            # 成行注文
            if sellMarketExit > 0:
                ClosePrice = Open[n]
                sellMarketExit = 0
            # 注文執行
            if ClosePrice > 0 and Volume[n]>sellCloseSize:
                if sellExecLot > sellCloseSize:
                    sell_exit_lot = sellCloseSize
                    sell_exec_price = sellExecPrice
                    sellExecLot = sellExecLot - sell_exit_lot
                else:
                    sell_exit_lot = sellExecLot
                    sell_exec_price = sellExecPrice
                    sellExecPrice = sellExecLot = 0
                ClosePrice = ClosePrice + spread + slippage
                ShortTrade[n] = ClosePrice #売りポジションクローズ
                ShortPL[n] = (sell_exec_price - ClosePrice) * sell_exit_lot #損益確定
                ShortPct[n] = ShortPL[n] / sell_exec_price

        # 利確 or 損切によるポジションの決済(エントリーと同じ足で決済しない)
        if buyExecPrice > 0 and not BuyNow and not OrderReject:
            ClosePrice = 0
            if stop_loss > 0:
                # 損切判定
                StopPrice = buyExecPrice - stop_loss
                if Low[n] <= StopPrice:
                    ClosePrice = Close[n]
            if take_profit > 0:
                # 利確判定
                LimitPrice = buyExecPrice + take_profit
                if High[n] >= LimitPrice:
                    ClosePrice = Close[n]
            if ClosePrice > 0 and Volume[n]>buyExecLot:
                ClosePrice = ClosePrice - slippage
                LongTrade[n] = ClosePrice #買いポジションクローズ
                LongPL[n] = (ClosePrice - buyExecPrice) * buyExecLot #損益確定
                LongPct[n] = LongPL[n] / buyExecPrice
                buyExecPrice = buyExecLot = 0

        if sellExecPrice > 0 and not SellNow and not OrderReject:
            ClosePrice = 0
            if stop_loss > 0:
                # 損切判定
                StopPrice = sellExecPrice + stop_loss
                if High[n] >= StopPrice:
                    ClosePrice = Close[n]
            if take_profit > 0:
                # 利確判定
                LimitPrice = sellExecPrice - take_profit
                if Low[n] <= LimitPrice:
                    ClosePrice = Close[n]
            if ClosePrice > 0 and Volume[n]>sellExecLot:
                ClosePrice = ClosePrice + slippage
                ShortTrade[n] = ClosePrice #売りポジションクローズ
                ShortPL[n] = (sellExecPrice - ClosePrice) * sellExecLot #損益確定
                ShortPct[n] = ShortPL[n] / sellExecPrice
                sellExecPrice = sellExecLot = 0

        capital = capital + ShortPL[n] + LongPL[n]
        max_profit = max(capital, max_profit)
        dd = max_profit - capital
        if max_drawdown>0 and dd>max_drawdown:
            dd_wait = wait_n_for_mdd
            max_profit = capital
        PositionSize[n] = buyExecLot - sellExecLot

    # ポジションクローズ
    if buyExecPrice > 0:
        ClosePrice = Close[N-1]
        LongTrade[N-1] = ClosePrice #買いポジションクローズ
        LongPL[N-1] = (ClosePrice - buyExecPrice) * buyExecLot #損益確定
        LongPct[N-1] = LongPL[N-1] / buyExecPrice
    if sellExecPrice > 0:
        ClosePrice = Close[N-1]
        ShortTrade[N-1] = ClosePrice #売りポジションクローズ
        ShortPL[N-1] = (sellExecPrice - ClosePrice) * sellExecLot #損益確定
        ShortPct[N-1] = ShortPL[N-1] / sellExecPrice

def BacktestCore2(Open, High, Low, Close, Volume, Delay, N, YourLogic,
                  LongTrade, LongPL, LongPct, ShortTrade, ShortPL, ShortPct, PositionSize,
                  max_delay_n):

    class Strategy:
        def __init__(self):
            self.positions = None
            self.position_size = 0
            self.position_avg_price = 0
            self.netprofit = 0
            self.open_orders = None
            self.orders = {}

        def order(self, myid, side, qty, limit = 0):
            self.orders[myid] = (-1 if side=='sell' else +1, limit, qty, myid)

        def cancel(self, myid):
            self.orders[myid] = (0, 0, 0, myid)

    positions = deque()
    position_size = 0
    position_avg_price = 0
    netprofit = np.zeros(N)
    open_orders = {}
    new_orders = []
    strategy = Strategy()

    for n in range(max_delay_n, N):

        # OHLCV取得
        O, H, L, C, V = Open[n], High[n], Low[n], Close[n], Volume[n]

        # 新規注文受付
        for o in new_orders:
            open_orders.update(o)
        new_orders = []

        # サイズ0の注文はキャンセル
        open_orders = {k:v for k,v in open_orders.items() if v[2]>0}

        # 約定判定（成行と指値のみ対応/現在の足で約定）
        executions = [o for o in open_orders.values() if (o[1]==0) or (o[1]>0 and ((o[0]<0 and H>o[1]) or (o[0]>0 and L<o[1])))]

        # 約定した注文を削除
        for e in executions:
            del open_orders[e[3]]

        # 約定処理
        for e in executions:
            o_side, exec_price, o_size, o_id = e
            exec_price = exec_price if exec_price>0 else O

            # 注文情報保存
            if o_side > 0:
                LongTrade[n] = exec_price
            else:
                ShortTrade[n] = exec_price

            # ポジション追加
            positions.append((o_side, exec_price, o_size))
            # print(n, 'Exec', o_id, o_side, exec_price, o_size)

            # 決済
            while len(positions)>=2:
                if positions[0][0] != positions[-1][0]:
                    l_side, l_price, l_size = positions.popleft()
                    r_side, r_price, r_size = positions.pop()
                    if l_size >= r_size:
                        pnl = (r_price - l_price) * (r_size * l_side)
                        l_size = round(l_size-r_size,8)
                        if l_size > 0:
                            positions.appendleft((l_side,l_price,l_size))
                        # print(n, 'Close', l_side, l_price, r_size, r_price, pnl)
                    else:
                        pnl = (r_price - l_price) * (l_size * l_side)
                        r_size = round(r_size-l_size,8)
                        if r_size > 0:
                            positions.append((r_side,r_price,r_size))
                        # print(n, 'Close', l_side, l_price, l_size, r_price, pnl)
                    # 決済情報保存
                    if l_side > 0:
                        LongPL[n] = LongPL[n] + pnl
                        LongPct[n] = LongPL[n] / r_price
                    else:
                        ShortPL[n] = ShortPL[n] + pnl
                        ShortPct[n] = ShortPL[n] / r_price
                else:
                    break

        # ポジションサイズ計算
        if len(positions):
            position_size = math.fsum(p[2] for p in positions)
            position_avg_price = math.fsum(p[1]*p[2] for p in positions) / position_size
            position_size = position_size * positions[0][0]
        else:
            position_size = position_avg_price = 0
        # print(n,'Pos',position_avg_price,position_size)

        # ポジション情報保存
        PositionSize[n] = position_size

        # 合計損益
        netprofit[n] = netprofit[n-1] + LongPL[n] + ShortPL[n]

        # 注文作成
        prev_n = n-Delay[n]
        strategy.positions, strategy.position_size, strategy.position_avg_price, strategy.netprofit, strategy.open_orders, strategy.orders = \
            positions, PositionSize[prev_n], position_avg_price, netprofit[prev_n], open_orders, {}
        YourLogic(Open[prev_n], High[prev_n], Low[prev_n], Close[prev_n], prev_n, strategy)
        new_orders.append(strategy.orders)

    # 残ポジションクローズ
    if len(positions):
        position_size = math.fsum(p[2]*p[0] for p in positions)
        position_avg_price = math.fsum(p[1]*p[2]*p[0] for p in positions)/position_size
        price = Close[N-1]
        pnl = (price - position_avg_price) * position_size
        if position_size > 0:
            # print(N-1, 'Close', 1, position_avg_price, position_size, price, pnl)
            LongPL[N-1] = pnl
            LongTrade[N-1] = price
        elif position_size < 0:
            # print(N-1, 'Close', -1, position_avg_price, position_size, price, pnl)
            ShortPL[N-1] = pnl
            ShortTrade[N-1] = price


def Backtest(ohlcv,
    buy_entry=None, sell_entry=None, buy_exit=None, sell_exit=None,
    stop_buy_entry=None, stop_sell_entry=None, stop_buy_exit=None, stop_sell_exit=None,
    limit_buy_entry=None, limit_sell_entry=None, limit_buy_exit=None, limit_sell_exit=None,
    buy_size=1.0, sell_size=1.0, max_buy_size=1.0, max_sell_size=1.0,
    spread=0, take_profit=0, stop_loss=0, trailing_stop=0, slippage=0, percent_of_equity=0.0, initial_capital=0.0,
    trades_per_second = 0, delay_n = 0, order_restrict_n = 0, max_drawdown=0, wait_seconds_for_mdd=0, yourlogic=None,
    bitflyer_rounding = True,
    **kwargs):
    Open = ohlcv.open.values #始値
    Low = ohlcv.low.values #安値
    High = ohlcv.high.values #高値
    Close = ohlcv.close.values #始値
    Volume = ohlcv.volume.values #出来高

    N = len(ohlcv) #データサイズ
    buyExecPrice = sellExecPrice = 0.0 # 売買価格
    buyStopEntry = buyStopExit = sellStopEntry = sellStopExit = 0
    buyExecLot = sellExecLot = 0

    LongTrade = np.zeros(N) # 買いトレード情報
    ShortTrade = np.zeros(N) # 売りトレード情報

    LongPL = np.zeros(N) # 買いポジションの損益
    ShortPL = np.zeros(N) # 売りポジションの損益

    LongPct = np.zeros(N) # 買いポジションの損益率
    ShortPct = np.zeros(N) # 売りポジションの損益率

    PositionSize = np.zeros(N) # ポジション情報

    place_holder = np.zeros(N) # プレースホルダ
    bool_place_holder = np.zeros(N, dtype=np.bool) # プレースホルダ
    if isinstance(buy_size, pd.Series):
        buy_size = buy_size.values
    else:
        buy_size = np.full(shape=(N), fill_value=float(buy_size))
    if isinstance(sell_size, pd.Series):
        sell_size = sell_size.values
    else:
        sell_size = np.full(shape=(N), fill_value=float(sell_size))

    buy_entry = bool_place_holder if buy_entry is None else buy_entry.values
    sell_entry = bool_place_holder if sell_entry is None else sell_entry.values
    buy_exit = bool_place_holder if buy_exit is None else buy_exit.values
    sell_exit = bool_place_holder if sell_exit is None else sell_exit.values

    # トレーリングストップ価格を設定(STOP注文として処理する)
    if trailing_stop > 0:
        stop_buy_exit = ohlcv.high - trailing_stop
        stop_sell_exit = ohlcv.low + trailing_stop

    stop_buy_entry = place_holder if stop_buy_entry is None else stop_buy_entry.values
    stop_sell_entry = place_holder if stop_sell_entry is None else stop_sell_entry.values
    stop_buy_exit = place_holder if stop_buy_exit is None else stop_buy_exit.values
    stop_sell_exit = place_holder if stop_sell_exit is None else stop_sell_exit.values

    limit_buy_entry = place_holder if limit_buy_entry is None else limit_buy_entry.values
    limit_sell_entry = place_holder if limit_sell_entry is None else limit_sell_entry.values
    limit_buy_exit = place_holder if limit_buy_exit is None else limit_buy_exit.values
    limit_sell_exit = place_holder if limit_sell_exit is None else limit_sell_exit.values

    timerange = (ohlcv.index[1] - ohlcv.index[0]).total_seconds()

    # 約定数
    if 'trades' in ohlcv:
        Trades = ohlcv.trades.values
    else:
        Trades = place_holder
    trades_per_n = trades_per_second * timerange

    # 配信遅延
    if 'delay' in ohlcv:
        Delay = ((ohlcv.delay+timerange/2)//timerange).values
        max_delay_n = np.max(Delay)
    else:
        Delay = np.full(shape=(N), fill_value=int(delay_n))
        max_delay_n = delay_n

    # ドローダウン時の待ち時間
    wait_n_for_mdd = math.ceil(wait_seconds_for_mdd / timerange)

    if yourlogic:
        # import line_profiler
        # lp = line_profiler.LineProfiler()
        # lp.add_function(BacktestCore2)
        # lp.add_function(yourlogic)
        # lp.enable()
        BacktestCore2(Open.astype(float), High.astype(float), Low.astype(float), Close.astype(float), Volume.astype(float), Delay.astype(int), N, yourlogic,
        LongTrade, LongPL, LongPct, ShortTrade, ShortPL, ShortPct, PositionSize, int(max_delay_n+1))
        # lp.disable()
        # lp.print_stats()
    else:
        BacktestCore(Open.astype(float), High.astype(float), Low.astype(float), Close.astype(float), Volume.astype(float), Trades.astype(int), N,
            buy_entry, sell_entry, buy_exit, sell_exit,
            stop_buy_entry.astype(float), stop_sell_entry.astype(float), stop_buy_exit.astype(float), stop_sell_exit.astype(float),
            limit_buy_entry.astype(float), limit_sell_entry.astype(float), limit_buy_exit.astype(float), limit_sell_exit.astype(float),
            buy_size, sell_size, max_buy_size, max_sell_size,
            float(spread), float(take_profit), float(stop_loss), float(trailing_stop), float(slippage), float(percent_of_equity), float(initial_capital),
            int(trades_per_n), int(delay_n+1), int(order_restrict_n), float(max_drawdown), int(wait_n_for_mdd),
            LongTrade, LongPL, LongPct, ShortTrade, ShortPL, ShortPct, PositionSize)

    if bitflyer_rounding:
        LongPL = LongPL.round()
        ShortPL = ShortPL.round()

    return BacktestReport(pd.DataFrame({
        'LongTrade':LongTrade, 'ShortTrade':ShortTrade,
        'LongPL':LongPL, 'ShortPL':ShortPL,
        'LongPct':LongPct, 'ShortPct':ShortPct,
        'PositionSize':PositionSize,
        }, index=ohlcv.index))


class BacktestReport:
    def __init__(self, DataFrame):
        self.DataFrame = DataFrame

        # ロング統計
        LongPL = DataFrame['LongPL']
        self.Long = dotdict()
        self.Long.PL = LongPL
        self.Long.Pct = DataFrame['LongPct']
        self.Long.Trades = np.count_nonzero(DataFrame['LongTrade'])
        if self.Long.Trades > 0:
            self.Long.GrossProfit = LongPL.clip_lower(0).sum()
            self.Long.GrossLoss =  LongPL.clip_upper(0).sum()
            self.Long.Profit = self.Long.GrossProfit + self.Long.GrossLoss
            self.Long.AvgReturn = self.Long.Pct[self.Long.Pct!=0].mean()
        else:
            self.Long.GrossProfit = 0.0
            self.Long.GrossLoss = 0.0
            self.Long.Profit = 0.0
            self.Long.AvgReturn = 0.0
        self.Long.WinTrades = np.count_nonzero(LongPL.clip_lower(0))
        if self.Long.WinTrades > 0:
            self.Long.WinMax = LongPL.max()
            self.Long.WinAverage = self.Long.GrossProfit / self.Long.WinTrades
            self.Long.WinPct = self.Long.Pct[self.Long.Pct > 0].mean()
        else:
            self.Long.WinMax = 0.0
            self.Long.WinAverage = 0.0
            self.Long.WinPct = 0.0
        self.Long.LossTrades = np.count_nonzero(LongPL.clip_upper(0))
        if self.Long.LossTrades > 0:
            self.Long.LossMax = LongPL.min()
            self.Long.LossAverage = self.Long.GrossLoss / self.Long.LossTrades
            self.Long.LossPct = self.Long.Pct[self.Long.Pct < 0].mean()
        else:
            self.Long.LossMax = 0.0
            self.Long.LossAverage = 0.0
            self.Long.LossPct = 0.0
        trades = self.Long.WinTrades+self.Long.LossTrades
        self.Long.WinRatio = self.Long.WinTrades / trades if trades else 0
        self.Long.EvenTrades = self.Long.Trades - trades

        # ショート統計
        ShortPL = DataFrame['ShortPL']
        self.Short = dotdict()
        self.Short.PL = ShortPL
        self.Short.Pct = DataFrame['ShortPct']
        self.Short.Trades = np.count_nonzero(DataFrame['ShortTrade'])
        if self.Short.Trades > 0:
            self.Short.GrossProfit = ShortPL.clip_lower(0).sum()
            self.Short.GrossLoss = ShortPL.clip_upper(0).sum()
            self.Short.Profit = self.Short.GrossProfit + self.Short.GrossLoss
            self.Short.AvgReturn = self.Short.Pct[self.Short.Pct!=0].mean()
        else:
            self.Short.GrossProfit = 0.0
            self.Short.GrossLoss = 0.0
            self.Short.Profit = 0.0
            self.Short.AvgReturn = 0.0
        self.Short.WinTrades = np.count_nonzero(ShortPL.clip_lower(0))
        if self.Short.WinTrades > 0:
            self.Short.WinMax = ShortPL.max()
            self.Short.WinAverage = self.Short.GrossProfit / self.Short.WinTrades
            self.Short.WinPct = self.Short.Pct[self.Short.Pct > 0].mean()
        else:
            self.Short.WinMax = 0.0
            self.Short.WinAverage = 0.0
            self.Short.WinPct = 0.0
        self.Short.LossTrades = np.count_nonzero(ShortPL.clip_upper(0))
        if self.Short.LossTrades > 0:
            self.Short.LossMax = ShortPL.min()
            self.Short.LossAverage = self.Short.GrossLoss / self.Short.LossTrades
            self.Short.LossPct = self.Short.Pct[self.Short.Pct < 0].mean()
        else:
            self.Short.LossMax = 0.0
            self.Short.LossTrades = 0.0
            self.Short.LossPct = 0.0
        trades = self.Short.WinTrades+self.Short.LossTrades
        self.Short.WinRatio = self.Short.WinTrades / trades if trades else 0
        self.Short.EvenTrades = self.Short.Trades - trades

        # 資産
        self.Equity = (LongPL + ShortPL).cumsum()

        # 全体統計
        self.All = dotdict()
        self.All.Trades = self.Long.Trades + self.Short.Trades
        self.All.WinTrades = self.Long.WinTrades + self.Short.WinTrades
        self.All.LossTrades = self.Long.LossTrades + self.Short.LossTrades
        trades = self.All.WinTrades+self.All.LossTrades
        self.All.EvenTrades = self.All.Trades - trades
        self.All.WinPct = (self.Long.WinPct + self.Short.WinPct) / 2
        self.All.WinRatio = self.All.WinTrades / trades if trades > 0 else 0
        self.All.GrossProfit = self.Long.GrossProfit + self.Short.GrossProfit
        self.All.GrossLoss = self.Long.GrossLoss + self.Short.GrossLoss
        self.All.WinAverage = self.All.GrossProfit / self.All.WinTrades if self.All.WinTrades > 0 else 0
        self.All.LossPct = (self.Long.LossPct + self.Short.LossPct) / 2
        self.All.LossAverage = self.All.GrossLoss / self.All.LossTrades if self.All.LossTrades > 0 else 0
        self.All.Profit = self.All.GrossProfit + self.All.GrossLoss
        self.All.AvgReturn = (self.Long.AvgReturn + self.Short.AvgReturn) / 2
        self.All.DrawDown = (self.Equity.cummax() - self.Equity).max()
        self.All.ProfitFactor = self.All.GrossProfit / -self.All.GrossLoss if -self.All.GrossLoss > 0 else 0
        if self.All.Trades > 1:
            pct = pd.concat([self.Long.Pct, self.Short.Pct])
            pct = pct[pct > 0]
            self.All.SharpeRatio = pct.mean() / pct.std()
        else:
            self.All.SharpeRatio = 1.0
        self.All.RecoveryFactor = self.All.ProfitFactor / self.All.DrawDown if self.All.DrawDown > 0 else 0
        self.All.ExpectedProfit = (self.All.WinAverage * self.All.WinRatio) + ((1 - self.All.WinRatio) * self.All.LossAverage)
        self.All.ExpectedValue = (self.All.WinRatio * (self.All.WinAverage / abs(self.All.LossAverage))) - (1 - self.All.WinRatio) if self.All.LossAverage < 0 else 1


    def __str__(self):
        return 'Long\n' \
        '  Trades :' + str(self.Long.Trades) + '\n' \
        '  EvenTrades :' + str(self.Long.EvenTrades) + '\n' \
        '  WinTrades :' + str(self.Long.WinTrades) + '\n' \
        '  WinMax :' + str(self.Long.WinMax) + '\n' \
        '  WinAverage :' + str(self.Long.WinAverage) + '\n' \
        '  WinPct :' + str(self.Long.WinPct) + '\n' \
        '  WinRatio :' + str(self.Long.WinRatio) + '\n' \
        '  LossTrades :' + str(self.Long.LossTrades) + '\n' \
        '  LossMax :' + str(self.Long.LossMax) + '\n' \
        '  LossAverage :' + str(self.Long.LossAverage) + '\n' \
        '  LossPct :' + str(self.Long.LossPct) + '\n' \
        '  GrossProfit :' + str(self.Long.GrossProfit) + '\n' \
        '  GrossLoss :' + str(self.Long.GrossLoss) + '\n' \
        '  Profit :' + str(self.Long.Profit) + '\n' \
        '  AvgReturn :' + str(self.Long.AvgReturn) + '\n' \
        '\nShort\n' \
        '  Trades :' + str(self.Short.Trades) + '\n' \
        '  EvenTrades :' + str(self.Short.EvenTrades) + '\n' \
        '  WinTrades :' + str(self.Short.WinTrades) + '\n' \
        '  WinMax :' + str(self.Short.WinMax) + '\n' \
        '  WinAverage :' + str(self.Short.WinAverage) + '\n' \
        '  WinPct :' + str(self.Short.WinPct) + '\n' \
        '  WinRatio :' + str(self.Short.WinRatio) + '\n' \
        '  LossTrades :' + str(self.Short.LossTrades) + '\n' \
        '  LossMax :' + str(self.Short.LossMax) + '\n' \
        '  LossAverage :' + str(self.Short.LossAverage) + '\n' \
        '  LossPct :' + str(self.Short.LossPct) + '\n' \
        '  GrossProfit :' + str(self.Short.GrossProfit) + '\n' \
        '  GrossLoss :' + str(self.Short.GrossLoss) + '\n' \
        '  Profit :' + str(self.Short.Profit) + '\n' \
        '  AvgReturn :' + str(self.Short.AvgReturn) + '\n' \
        '\nAll\n' \
        '  Trades :' + str(self.All.Trades) + '\n' \
        '  EvenTrades :' + str(self.All.EvenTrades) + '\n' \
        '  WinTrades :' + str(self.All.WinTrades) + '\n' \
        '  WinAverage :' + str(self.All.WinAverage) + '\n' \
        '  WinPct :' + str(self.All.WinPct) + '\n' \
        '  WinRatio :' + str(self.All.WinRatio) + '\n' \
        '  LossTrades :' + str(self.All.LossTrades) + '\n' \
        '  LossAverage :' + str(self.All.LossAverage) + '\n' \
        '  LossPct :' + str(self.All.LossPct) + '\n' \
        '  GrossProfit :' + str(self.All.GrossProfit) + '\n' \
        '  GrossLoss :' + str(self.All.GrossLoss) + '\n' \
        '  Profit :' + str(self.All.Profit) + '\n' \
        '  AvgReturn :' + str(self.All.AvgReturn) + '\n' \
        '  DrawDown :' + str(self.All.DrawDown) + '\n' \
        '  ProfitFactor :' + str(self.All.ProfitFactor) + '\n' \
        '  SharpeRatio :' + str(self.All.SharpeRatio) + '\n'

# 参考
# https://qiita.com/kenchin110100/items/ac3edb480d789481f134

def BacktestIteration(testfunc, default_parameters, hyperopt_parameters, max_evals, maximize=lambda r:r.All.Profit):

    needs_header = [True]

    def go(args):
        params = default_parameters.copy()
        params.update(args)
        report = testfunc(**params)
        result = {}
        for k,v in params.items():
            if not isinstance(v, pd.DataFrame):
                result[k] = v
        result.update(report.All)
        if needs_header[0]:
            print(','.join(result.keys()))
        print(','.join([str(x) for x in result.values()]))
        needs_header[0] = False
        return report

    if max_evals > 0:
        # 試行の過程を記録するインスタンス
        trials = Trials()

        best = fmin(
            # 最小化する値を定義した関数
            lambda args: -1 * maximize(go(args)),
            # 探索するパラメータのdictもしくはlist
            hyperopt_parameters,
            # どのロジックを利用するか、基本的にはtpe.suggestでok
            # rand.suggest ランダム・サーチ？
            # anneal.suggest 焼きなましっぽい
            algo=tpe.suggest,
            #algo=rand.suggest,
            #algo=anneal.suggest,
            max_evals=max_evals,
            trials=trials,
            # 試行の過程を出力
            verbose=0
        )
    else:
        best = default_parameters

    params = default_parameters.copy()
    params.update(best)
    report = go(params)
    print(report)
    return (params, report)


if __name__ == '__main__':

    from utils import stop_watch

    ohlcv = pd.read_csv('csv/bitmex_2018_1m.csv', index_col='timestamp', parse_dates=True)
    buy_entry = ohlcv.close > ohlcv.close.shift(1)
    sell_entry = ohlcv.close < ohlcv.close.shift(1)
    buy_exit = sell_entry
    sell_exit = buy_entry
    Backtest = stop_watch(Backtest)

    Backtest(**locals())
    Backtest(**locals())
    Backtest(**locals())
    Backtest(**locals())
