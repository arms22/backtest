# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from numba import jit, b1, f8, i8, void
from .utils import dotdict
from hyperopt import hp, tpe, Trials, fmin, rand, anneal, STATUS_OK
from collections import deque, defaultdict
from operator import itemgetter

# PythonでFXシストレのバックテスト(1)
# https://qiita.com/toyolab/items/e8292d2f051a88517cb2 より

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
            if buyLimitEntry > 0 and Low[n] < buyLimitEntry:
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
            if buyLimitExit > 0 and High[n] > buyLimitExit:
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
            if sellLimitEntry > 0 and High[n] > sellLimitEntry:
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
            if sellLimitExit > 0 and Low[n] < sellLimitExit:
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

BacktestSettlements = []
def BacktestCore2(Open, High, Low, Close, Bid, Ask, BuyVolume, SellVolume, Trades, Timestamp, Delay, N, YourLogic, EntryTiming,
                  LongTrade, LongPL, LongPct, ShortTrade, ShortPL, ShortPct, PositionSize, NumberOfRequests, NumberOfOrders):
    EXPIRE_MAX = np.finfo(np.float64).max
    class Strategy:
        def __init__(self):
            self.positions = []
            self.position_size = 0
            self.position_avg_price = 0
            self.netprofit = 0
            self.active_orders = {}
            self.cancel_orders = {}
            self.new_orders = {}
            self.accept_orders = {}
            self.number_of_requests = 0
            self.number_of_orders = 0
            self.order_ref_id = 1
            self.order_ref_id_from = defaultdict(int)
            self.trailing_orders = {}
            self.settlements = []

        def order(self, myid, side, qty, limit=0, expire_at=EXPIRE_MAX):
            ref = self.order_ref_id_from[myid]
            if ref in self.active_orders:
                self.cancel_orders[ref] = {'size':0,'myid':myid}
                self.number_of_requests += 1
            if ref not in self.accept_orders:
                self.number_of_requests += 1
                self.number_of_orders += 1
                self.order_ref_id += 1
                order = {'side':1 if side=='buy' else -1, 'price':limit, 'size':qty, 'myid':myid, 'expire_at':expire_at}
                self.new_orders[self.order_ref_id] = order
                self.accept_orders[self.order_ref_id] = order
                self.order_ref_id_from[myid] = self.order_ref_id

        def close(self, myid, side, qty, limit=0, take_profit=None, stop_loss=None, tralling_stop=None, entryPrice=None, lastPrice=None):
            if lastPrice and entryPrice:
                if tralling_stop:
                    if myid in self.trailing_orders:
                        if self.trailing_orders[myid]['entryPrice'] != entryPrice:
                            del self.trailing_orders[myid]
                    tralling_order = self.trailing_orders.get(myid,{
                        'entryPrice':entryPrice,
                        'trallingPrice':entryPrice,
                    })
                    price_change = lastPrice-tralling_order['trallingPrice']
                    tralling_order['trallingPrice'] = lastPrice
                    if 'buy' == side:
                        if price_change>tralling_stop:
                            self.order(myid,side,qty,limit)
                        else:
                            self.cancel(myid)
                    else:
                        if price_change<-tralling_stop:
                            self.order(myid,side,qty,limit)
                        else:
                            self.cancel(myid)
                else:
                    if 'buy' == side:
                        pnl = entryPrice - lastPrice
                    else:
                        pnl = lastPrice - entryPrice
                    if take_profit and pnl>take_profit:
                        self.order(myid,side,qty,limit)
                    elif stop_loss and pnl<-stop_loss:
                        self.order(myid,side,qty,limit)
                    else:
                        self.cancel(myid)
            else:
                self.order(myid,side,qty,limit)

        def cancel(self, myid):
            ref = self.order_ref_id_from[myid]
            if ref in self.active_orders:
                self.cancel_orders[ref] = {'size':0,'myid':myid}
                self.number_of_requests += 1

        def get_order(self, myid):
            ref = self.order_ref_id_from[myid]
            return self.active_orders.get(ref,None) or self.accept_orders.get(ref,None)

        def get_orders(self):
            orders = list(self.active_orders.values())+list(o for o in self.accept_orders.values() if o['size']>0)
            buy_orders = sorted([o for o in orders if o['side']==1],key=itemgetter('price'))
            sell_orders = sorted([o for o in orders if o['side']==-1],key=itemgetter('price'))
            return buy_orders, sell_orders

        def cancel_order_all(self):
            for k,ref in self.order_ref_id_from.items():
                if ref in self.active_orders:
                    self.cancel_orders[ref] = {'size':0,'myid':k}
            self.number_of_requests += 1

        def close_position(self):
            if self.position_size>0:
                myid = '__Lc__'
                order = {'side':-1, 'price':0, 'size':self.position_size, 'myid':myid, 'expire_at':EXPIRE_MAX}
            elif self.position_size<0:
                myid = '__Sc__'
                order = {'side':1, 'price':0, 'size':-self.position_size, 'myid':myid, 'expire_at':EXPIRE_MAX}
            else:
                order = None
            if order is not None:
                ref = self.order_ref_id_from[myid]
                if ref not in self.accept_orders:
                    self.number_of_requests += 1
                    self.number_of_orders += 1
                    self.order_ref_id += 1
                    self.new_orders[self.order_ref_id] = order
                    self.accept_orders[self.order_ref_id] = order
                    self.order_ref_id_from[myid] = self.order_ref_id

    positions = deque()
    position_size = 0
    position_avg_price = 0
    netprofit = 0
    open_orders = {}
    accept_orders = []
    strategy = Strategy()
    BacktestSettlements.clear()

    for n in range(1, N-1):

        # OHLCV取得
        O, H, L, C, bid, ask, bv, sv, T, CanEntry = Open[n], High[n], Low[n], Close[n], Bid[n], Ask[n], BuyVolume[n], SellVolume[n], Timestamp[n], EntryTiming[n]

        # 注文受付
        if len(accept_orders):
            remaining, exec_t = [], T-Delay[n]
            for accept_t, o in accept_orders:
                if exec_t>accept_t:
                    open_orders.update(o)
                else:
                    remaining.append((accept_t,o))

            # 残った注文は次の足で処理
            accept_orders = remaining

        if len(open_orders):

            # サイズ0の注文・期限切れの注文キャンセル
            open_orders = {k:o for k,o in open_orders.items() if o['size']>0 and T<o['expire_at']}

            # 約定判定（成行と指値のみ対応/現在の足で約定）
            executions = {k:o for k,o in open_orders.items() if (o['price']==0) or\
                ((o['side']<0 and H>o['price'] and bv>0) or (o['side']>0 and L<o['price'] and sv>0))}
            # if C>O:
            #     executions = {k:o for k,o in open_orders.items() if (o['price']==0) or\
            #         ((o['side']<0 and H>o['price'] and bv>0) or (o['side']>0 and (O+L)/2<o['price'] and sv>0))}
            # elif C<O:
            #     executions = {k:o for k,o in open_orders.items() if (o['price']==0) or\
            #         ((o['side']<0 and (H+O)/2>o['price'] and bv>0) or (o['side']>0 and L<o['price'] and sv>0))}
            # else:
            #     executions = {k:o for k,o in open_orders.items() if (o['price']==0) or\
            #         ((o['side']<0 and H>o['price'] and bv>0) or (o['side']>0 and L<o['price'] and sv>0))}

            # 約定処理
            for k,e in executions.items():
                o_side, o_price, o_size, o_id = e['side'], e['price'], e['size'], e['myid']

                # 約定価格とサイズ
                if o_price>0:
                    if o_side>0:
                        exec_price = o_price
                        exec_size = min(o_size, sv)
                    else:
                        exec_price = o_price
                        exec_size = min(o_size, bv)
                else:
                    if o_side>0:
                        exec_price = ask
                        exec_size = o_size
                    else:
                        exec_price = bid
                        exec_size = o_size
                # print(n,'Exec',o_side,exec_price,exec_size,o_id)

                # 部分約定なら残サイズを戻す
                if exec_size < o_size:
                    open_orders[k]['size'] = o_size-exec_size
                else:
                    del open_orders[k]

                if exec_size>0:
                    # 注文情報保存
                    if o_side > 0:
                        LongTrade[n] = exec_price
                    else:
                        ShortTrade[n] = exec_price

                    # ポジション追加
                    positions.append((o_side, exec_price, exec_size, T))

                    # 決済
                    while len(positions)>=2:
                        if positions[0][0] != positions[-1][0]:
                            l_side, l_price, l_size, l_T = positions.popleft()
                            r_side, r_price, r_size, r_T = positions.pop()
                            if l_size >= r_size:
                                pnl = (r_price - l_price) * (r_size * l_side)
                                size = r_size
                                order_remaing = round(l_size-r_size,8)
                                if order_remaing > 0:
                                    positions.appendleft((l_side,l_price,order_remaing,l_T))
                                # print(n, 'Close', l_side, l_price, r_size, r_price, pnl)
                            else:
                                pnl = (r_price - l_price) * (l_size * l_side)
                                size = l_size
                                order_remaing = round(r_size-l_size,8)
                                if order_remaing > 0:
                                    positions.append((r_side,r_price,order_remaing,r_T))
                                # print(n, 'Close', l_side, l_price, l_size, r_price, pnl)
                            # 決済情報保存
                            if l_side > 0:
                                LongPL[n] = LongPL[n]+pnl
                                LongPct[n] = LongPL[n]/r_price
                            else:
                                ShortPL[n] = ShortPL[n]+pnl
                                ShortPct[n] = ShortPL[n]/r_price
                            strategy.settlements.append({'side':l_side, 'price':r_price, 'size':size, 'pnl':pnl, 'entry_time':l_T, 'exit_time':T})
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

                    # 合計損益
                    netprofit = netprofit + LongPL[n] + ShortPL[n]

        # ポジション情報保存
        PositionSize[n] = position_size

        # 注文作成
        if CanEntry:
            strategy.positions, strategy.position_size, strategy.position_avg_price, strategy.netprofit, \
                strategy.active_orders, strategy.new_orders, strategy.cancel_orders, strategy.accept_orders = \
                    list(positions), position_size, position_avg_price, netprofit,\
                        open_orders, {}, {}, {k:v for t,o in accept_orders for k,v in o.items()}

            YourLogic(O, H, L, C, n, strategy)

            # 注文
            # accept_orders.append((T+0.08,strategy.cancel_orders))
            # accept_orders.append((T+0.06,strategy.new_orders))
            accept_orders.append((T+Delay[n]+0.08,strategy.cancel_orders))
            accept_orders.append((T+Delay[n]+0.06,strategy.new_orders))

        # API発行回数・新規注文数保存
        NumberOfRequests[n] = strategy.number_of_requests
        NumberOfOrders[n] = strategy.number_of_orders

    # 残ポジションクローズ
    if len(positions):
        position_size = math.fsum(p[2] for p in positions)
        position_avg_price = math.fsum(p[1]*p[2] for p in positions) / position_size
        position_size = position_size * positions[0][0]
        price = Close[N-1]
        pnl = (price - position_avg_price) * position_size
        if position_size > 0:
            # print(N-1, 'Close', 1, position_avg_price, position_size, price, pnl)
            ShortTrade[N-1] = price
            LongPL[N-1] = pnl
            LongPct[N-1] = LongPL[N-1]/price
        elif position_size < 0:
            # print(N-1, 'Close', -1, position_avg_price, position_size, price, pnl)
            LongTrade[N-1] = price
            ShortPL[N-1] = pnl
            ShortPct[N-1] = ShortPL[N-1]/price
    BacktestSettlements.extend(strategy.settlements)

def Backtest(ohlcv,
    buy_entry=None, sell_entry=None, buy_exit=None, sell_exit=None,
    stop_buy_entry=None, stop_sell_entry=None, stop_buy_exit=None, stop_sell_exit=None,
    limit_buy_entry=None, limit_sell_entry=None, limit_buy_exit=None, limit_sell_exit=None,
    buy_size=1.0, sell_size=1.0, max_buy_size=1.0, max_sell_size=1.0,
    spread=0, take_profit=0, stop_loss=0, trailing_stop=0, slippage=0, percent_of_equity=0.0, initial_capital=0.0,
    trades_per_second = 0, delay_n = 0, order_restrict_n = 0, max_drawdown=0, wait_seconds_for_mdd=0, yourlogic=None,
    bitflyer_rounding = False, interval_yourlogic = None, volume_yourlogic = None, outliers_sigma = 0,
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
    NumberOfRequests = np.zeros(N)
    NumberOfOrders = np.zeros(N)

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

    # 約定数
    if 'trades' in ohlcv:
        Trades = ohlcv.trades.values
    else:
        Trades = place_holder

    if yourlogic:
        if 'bid' in ohlcv:
            Bid = ohlcv.bid.values
        else:
            Bid = Open-spread/2
        if  'ask' in ohlcv:
            Ask = ohlcv.ask.values
        else:
            Ask = Open+spread/2
        if 'buy_volume' in ohlcv:
            BuyVolume = ohlcv.buy_volume.values
        else:
            BuyVolume = Volume/2
        if 'sell_volume' in ohlcv:
            SellVolume = ohlcv.sell_volume.values
        else:
            SellVolume = Volume/2

        # 基準時刻
        Timestamp = ohlcv.index.astype(np.int64) / 10**9

        # 遅延情報
        if 'delay' in ohlcv:
            Delay = ohlcv.delay.clip_lower(0).values
        else:
            Delay = np.full(shape=(N), fill_value=int(delay_n))

        # エントリータイミング
        if interval_yourlogic:
            ti = Timestamp // interval_yourlogic
            EntryTiming = ti - np.roll(ti, 1)
        else:
            if volume_yourlogic:
                vi = np.cumsum(ohlcv.volume.values) // volume_yourlogic
                EntryTiming = vi - np.roll(vi, 1)
            else:
                EntryTiming = np.full(shape=(N), fill_value=1)
        # import line_profiler
        # lp = line_profiler.LineProfiler()
        # lp.add_function(BacktestCore2)
        # lp.add_function(yourlogic)
        # lp.enable()
        BacktestCore2(Open.astype(float), High.astype(float), Low.astype(float), Close.astype(float), Bid.astype(float), Ask.astype(float),
            BuyVolume.astype(float), SellVolume.astype(float), Trades.astype(int), Timestamp.astype(float), Delay.astype(float), N, yourlogic, EntryTiming.astype(int),
            LongTrade, LongPL, LongPct, ShortTrade, ShortPL, ShortPct, PositionSize, NumberOfRequests, NumberOfOrders)
        # lp.disable()
        # lp.print_stats()
    else:
        # タイムフレーム
        timeframe = (ohlcv.index[-1] - ohlcv.index[0]).total_seconds()/N

        # 1フレームあたりの約定数
        trades_per_n = trades_per_second * timeframe

        # ドローダウン時の待ち時間
        wait_n_for_mdd = math.ceil(wait_seconds_for_mdd / timeframe)

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

    if outliers_sigma:
        sd = np.std(LongPL[LongPL!=0])
        LongPL[LongPL>sd*outliers_sigma] = 0
        sd = np.std(ShortPL[ShortPL!=0])
        ShortPL[ShortPL>sd*outliers_sigma] = 0

    return BacktestReport(pd.DataFrame({
        'LongTrade':LongTrade, 'ShortTrade':ShortTrade,
        'LongPL':LongPL, 'ShortPL':ShortPL,
        'LongPct':LongPct, 'ShortPct':ShortPct,
        'PositionSize':PositionSize,
        'NumberOfRequests':NumberOfRequests, 'NumberOfOrders':NumberOfOrders,
        }, index=ohlcv.index))


class BacktestReport:
    def __init__(self, DataFrame):
        self.DataFrame = DataFrame

        # API利用状況
        requests = DataFrame['NumberOfRequests'].diff()
        orders = DataFrame['NumberOfOrders'].diff()
        self.DataFrame['Requests/min'] = requests.rolling('1T').sum()
        self.DataFrame['Requests/5min'] = requests.rolling('5T').sum()
        self.DataFrame['Orders/min'] = orders.rolling('1T').sum()

        # ロング統計
        LongPL = DataFrame['LongPL']
        self.Long = dotdict()
        self.Long.PL = LongPL
        self.Long.Pct = DataFrame['LongPct']
        self.Long.Trades = np.count_nonzero(LongPL)
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
        self.Short.Trades = np.count_nonzero(ShortPL)
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
        self.DataFrame['Equity'] = self.Equity

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

def BacktestIteration(testfunc, default_parameters, hyperopt_parameters, max_evals, maximize=lambda r:r.All.Profit, verbose=False):

    once = True

    def go(args):
        nonlocal once
        params = default_parameters.copy()
        params.update(args)
        report = testfunc(**params)
        result = {k:v for k,v in params.items() if not isinstance(v, pd.DataFrame)}
        result.update(report.All)
        if verbose:
            if once:
                print(','.join(result.keys()))
                once = False
            print(','.join([str(x) for x in result.values()]))
        return {
            'loss':-1 * maximize(report),
            'status':STATUS_OK,
            'other_stuff':result,
        }

    # 試行の過程を記録するインスタンス
    trials = Trials()

    if max_evals>0:
        best = fmin(
            # 最小化する値を定義した関数
            go,
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
        params = default_parameters.copy()
        params.update(best)
        report = testfunc(**params)
        results = pd.DataFrame([r['other_stuff'] for r in trials.results])
    else:
        best = {k:default_parameters[k] for k in hyperopt_parameters.keys()}
        params = default_parameters.copy()
        params.update(best)
        report = testfunc(**params)
        result = {k:v for k,v in params.items() if not isinstance(v, pd.DataFrame)}
        result.update(report.All)
        results = pd.DataFrame([result])
    return best, report, results


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
