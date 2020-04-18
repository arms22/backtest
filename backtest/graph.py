# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def ohlcv_scatter(ohlcv):
    high_sca = go.Scatter(
        x = ohlcv.high.index,
        y = ohlcv.high,
        name = 'Hihg', mode = 'lines')
    low_sca = go.Scatter(
        x = ohlcv.low.index,
        y = ohlcv.low,
        name = 'Low', mode = 'lines')
    return high_sca, low_sca

def buy_executions_scatter(r):
    buy_exec = r.DataFrame.LongTrade[r.DataFrame.LongTrade>0]
    return go.Scatter(
        x = buy_exec.index,
        y = buy_exec,
        name = 'Buy', mode = 'markers')

def sell_executions_scatter(r):
    sell_exec = r.DataFrame.ShortTrade[r.DataFrame.ShortTrade>0]
    return go.Scatter(
        x = sell_exec.index,
        y = sell_exec,
        name = 'Sell', mode = 'markers')

def netprofit_scatter(r):
    netprofit = r.Equity[r.Equity.diff()!=0]
    netprofit_sca = go.Scatter(
        x = netprofit.index,
        y = netprofit,
        name = 'Net Profit', mode = 'lines')

    netprofit_cummax = netprofit.cummax()
    netprofit_cummax_sca = go.Scatter(
        x = netprofit_cummax.index,
        y = netprofit_cummax,
        name = 'Net Profit Max', mode = 'lines')

    drawdown = netprofit_cummax - netprofit
    drawdown_sca = go.Scatter(
        x = drawdown.index,
        y = drawdown.cummax(),
        name = 'Drawdown', mode = 'lines')
    return netprofit_sca, netprofit_cummax_sca, drawdown_sca

def positions_scatter(r):
    positions = r.DataFrame.PositionSize[r.DataFrame.PositionSize.diff()!=0]
    return go.Scatter(
        x = positions.index,
        y = positions,
        name = 'Positions', mode = 'lines', line_shape='hv')

def profit_factor_scatter(r, period=100, upper=10, lower=-10):
    tradepnl = r.DataFrame.LongPL+r.DataFrame.ShortPL
    tradepnl = tradepnl[tradepnl!=0]
    grossprofit = tradepnl.clip_lower(0)
    grossloss = -tradepnl.clip_upper(0)
    grossprofitsum = grossprofit.rolling(window=period,min_periods=10).sum().clip_lower(1)
    grosslosssum = grossloss.rolling(window=period,min_periods=10).sum().clip_lower(1)
    profitfactor = grossprofitsum / grosslosssum
    profitfactor = profitfactor.clip(upper,lower)
    return go.Scatter(
        x = profitfactor.index,
        y = profitfactor,
        name = 'Profit Factor', mode = 'lines')

def ReportFigure(r, title_text='', height=600, ohlcv=None):

    fig = make_subplots(
        # rows=4,
        rows=3,
        cols=1,
        shared_xaxes=True,
        # row_heights=[0.5,0.3,0.1,0.1],
        row_heights=[0.5,0.25,0.25],
        # vertical_spacing=0.02,
        vertical_spacing=0.03,
        # subplot_titles=('Executions and Net Profit','Positions','Profit Factor'),
        specs=[
            # [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}]])

    fig.update_layout(
        height=height,
        title_text=title_text)

    if ohlcv:
        high, low = ohlcv_scatter(ohlcv)
        fig.add_trace(high,row=1,col=1)
        fig.add_trace(low,row=1,col=1)

    fig.add_trace(buy_executions_scatter(r),row=1,col=1)
    fig.add_trace(sell_executions_scatter(r),row=1,col=1)

    netprofit, netprofit_cummax, drawdown = netprofit_scatter(r)
    # fig.add_trace(netprofit,row=2,col=1)
    # fig.add_trace(netprofit_cummax,row=2,col=1)
    # fig.add_trace(drawdown,row=2,col=1,secondary_y=True)
    fig.add_trace(netprofit,row=1,col=1,secondary_y=True)
    fig.add_trace(netprofit_cummax,row=1,col=1,secondary_y=True)

    # fig.add_trace(positions_scatter(r),row=3,col=1)
    fig.add_trace(positions_scatter(r),row=2,col=1)

    # fig.add_trace(profit_factor_scatter(r),row=4,col=1)
    fig.add_trace(profit_factor_scatter(r),row=3,col=1)

    return fig
