import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_spaghetti(filtered_data):
    symbols = filtered_data['symbol'].unique()
    for sym in symbols:
        mask = filtered_data['symbol'] == sym
        initial_close_price = filtered_data.loc[mask, 'close'].iloc[0]
        filtered_data.loc[mask, 'close_percentage'] = \
            (filtered_data.loc[mask, 'close'] / initial_close_price - 1) * 100

    fig = px.line(filtered_data, x='datetime', y='close_percentage', color='symbol', height=500,
                  labels={'close_percentage': 'Gains (%)'}
                  )
    fig.update_layout(
        title=f"Performance Lines, {filtered_data['tf'].unique()[0]}",
        xaxis_title='Datetime',
        yaxis_title='Gains (%)',
    )
    return fig


def plot_acceleration(filtered_data):
    modif_data = filtered_data.copy().groupby('symbol').tail(1)
    long_mask = \
        (modif_data['close'] > modif_data['hma55']) & \
        (modif_data['close'] > modif_data[['ema50', 'q_mean']].min(axis=1))
    short_mask = \
        (modif_data['close'] < modif_data['hma55']) & \
        (modif_data['close'] < modif_data[['ema50', 'q_mean']].max(axis=1))

    modif_data['signal'] = np.where(long_mask, 1, np.nan)
    modif_data.loc[short_mask, 'signal'] = 0
    modif_data['dist_perc'] = (modif_data['close'] - modif_data['q_mean']) / modif_data['close'] * 100

    signal_data = modif_data.loc[((modif_data['signal'] == 1) | (modif_data['signal'] == 0)) &
                                 ((modif_data['dist_perc'] < 10) & (modif_data['dist_perc'] > -10))]

    x_symbols = signal_data['symbol']
    y_val = signal_data['dist_perc']

    fig = px.scatter(signal_data, x=x_symbols, y=y_val,
                     symbol="signal", symbol_map={1: "triangle-up", 0: "triangle-down"},
                     color="dist_perc",
                     color_continuous_scale=px.colors.diverging.Picnic_r, color_continuous_midpoint=0,
                     labels={"signal": "short/long", "dist_perc": "dist[%]"})

    # Set the y-axis range from -10 to 10
    y_range = [-10, 10]
    fig.update_yaxes(range=y_range, tickmode='linear', dtick=2)
    # update x axis labels
    fig.update_xaxes(tickangle=-90, tickvals=x_symbols,
                     ticktext=[_sym.split("/")[0] for _sym in x_symbols])

    fig.update_traces(marker=dict(size=15))  # symbol size
    fig.update_layout(coloraxis_colorbar=dict(yanchor="top"),  # to avoid legend overflow
                      title=f"Acceleration, {filtered_data['tf'].unique()[0]}")

    for j in range(0, len(x_symbols)):  # plot vertical lines
        fig.add_shape(
            dict(
                type="line",
                x0=j,
                x1=j,
                y0=y_range[0],
                y1=y_range[1],
                line=dict(color="gray", width=1),
            )
        )
    fig.add_shape(
        dict(
            type="line",
            x0=0,
            x1=len(x_symbols) - 1,
            y0=0,
            y1=0,
            line=dict(color="white", width=2)
        )
    )
    return fig


def plot_acceleration2(filtered_data):
    modif_data = filtered_data.copy().groupby('symbol').tail(5)

    long_mask = \
        (modif_data['close'] > modif_data['hma55']) & \
        (modif_data['close'] > modif_data[['ema50', 'q_mean']].min(axis=1))
    short_mask = \
        (modif_data['close'] < modif_data['hma55']) & \
        (modif_data['close'] < modif_data[['ema50', 'q_mean']].max(axis=1))

    modif_data['signal'] = np.where(long_mask, 1, 0)
    modif_data.loc[short_mask, 'signal'] = -1
    modif_data['dist_perc'] = (modif_data['close'] - modif_data['q_mean']) / modif_data['close'] * 100

    symbols = modif_data.groupby('symbol').tail(1)
    symbols = symbols.loc[
        ((symbols['signal'] == 1) | (symbols['signal'] == -1)) &
        ((symbols['dist_perc'] < 10) & (symbols['dist_perc'] > -10))]['symbol']

    signal_data = modif_data.loc[modif_data['symbol'].isin(symbols)]
    signal_data.reset_index(inplace=True)
    del signal_data['index']

    # Create a new column 'x' with unique and continuously incremented x-coordinates
    signal_data['x'] = signal_data.index.values
    signal_data['x_val'] = 0
    x = 0
    space = 10
    j: int
    for j, row in signal_data.iterrows():
        if j % 5 == 0 and j >= 5:
            x += space
        signal_data.loc[j, 'x_val'] = row['x'] + x

    x_val = signal_data['x_val']

    # return None if x_val df is empty
    if x_val.empty:
        return None

    fig = go.Figure()
    for sym in symbols:
        for j in range(0, len(signal_data)):
            if sym == signal_data.iloc[j]['symbol']:
                fig.add_trace(
                    go.Scatter(
                        x=signal_data.loc[signal_data['symbol'] == sym]['x_val'].tolist(),
                        y=signal_data.loc[signal_data['symbol'] == sym]['dist_perc'].tolist(),
                        name=sym,
                        # mode="lines",
                        # line={"color": "red", 'colorscale': 'Rainbow'},
                        marker={'color': 'white'},
                        showlegend=False,
                    )
                )
    # Set the y-axis range from -10 to 10
    y_range = [-10, 10]
    fig.update_yaxes(range=y_range, dtick=2)
    # update x axis labels
    fig.update_xaxes(tickangle=-90)

    fig.update_traces(marker=dict(size=6))  # symbol size
    fig.update_layout(title=f"Acceleration, {filtered_data['tf'].unique()[0]}",
                      xaxis_title="symbol", yaxis_title="distance [%]",
                      xaxis=dict(
                          tickmode='array',
                          tickvals=[val for val in x_val if val % 5 == 0],
                          ticktext=[sym.split("/")[0] for sym in symbols]
                      ))
    fig.add_shape(
        dict(
            type="line",
            x0=0,
            x1=x_val.tolist()[-1],
            y0=0,
            y1=0,
            line=dict(color="red", width=2)
        ),

    )
    return fig


def plot_order_blocks(filtered_data):
    fig = go.Figure()

    for sym in filtered_data['symbol'].unique():
        sym_df = filtered_data.loc[filtered_data['symbol'] == sym]
        for j, row in sym_df.iterrows():
            dtime = row['datetime']
            s_d = row['s_d']
            color = 'gray'
            if s_d == 'D':
                color = 'green'
            elif s_d == 'S':
                color = 'red'
            fig.add_trace(go.Bar(y=[sym], x=[1], orientation='h', name=s_d, hovertemplate=str(dtime),
                                 marker={'color': color}, textangle=0, textfont=dict(size=8)))

    fig.update_traces(marker_line_color='black', marker_line_width=.8, opacity=1)

    # Customize the appearance
    height = 1000 if len(st.session_state['symbols']) < 50 else 1500
    fig.update_layout(
        # autosize=False,
        height=height,
        title=f"Order Blocks, {filtered_data['tf'].unique()[0]}",
        yaxis_title='Symbol',
        xaxis_title='Bars',
        barmode='stack',  # Stacks the bars on top of each other
        showlegend=False,  # Hide the color legend
    )
    return fig


def plot_map(filtered_data):
    fig = go.Figure()
    size = len(st.session_state['symbols'])

    # Create bars for each symbol and add each color with a width of 1
    for sym in st.session_state['symbols']:
        sym_df = filtered_data[(filtered_data['symbol'] == sym)]

        for j in range(len(sym_df)):
            bar_color = sym_df.iloc[j]['bar_color']
            dtime = sym_df.iloc[j]['datetime']
            k = sym_df.iloc[j]['stochrsi_k']

            fig.add_trace(go.Bar(y=[sym], x=[1], orientation='h', name=k, marker_color=bar_color,
                                 hovertemplate=str(dtime)))

    fig.update_traces(marker_line_color='black', marker_line_width=.8, opacity=1)

    # Customize the appearance
    if size < 20:
        height = 500
    elif size < 50:
        height = 1000
    else:
        height = 1500
    fig.update_layout(
        # autosize=False,
        height=height,
        title=f"StochRSI Map, {filtered_data['tf'].unique()[0]}",
        yaxis_title='Symbol',
        xaxis_title='Bars',
        barmode='stack',  # Stacks the bars on top of each other
        showlegend=False,  # Hide the color legend
    )
    return fig


@st.cache_data
def plot_adv_map(filtered_df, last_data):
    fig = go.Figure()
    size = len(st.session_state['symbols'])
    text_size = 8 if size < 20 else 5

    # Create stacked bar chart for each symbol, each bar segment has height of 1
    for sym in st.session_state['symbols']:
        sym_df = filtered_df[(filtered_df['symbol'] == sym)]
        for j in range(len(sym_df)):
            bar_color = sym_df.iloc[j]['bar_color']
            dtime = sym_df.iloc[j]['datetime']
            k = sym_df.iloc[j]['stochrsi_k']
            trade = sym_df.iloc[j]['trade']
            if not trade:
                display_text = ""
            elif trade == "L":
                display_text = "⬜"
            else:
                display_text = "⬛"

            fig.add_trace(go.Bar(y=[sym], x=[1], orientation='h', name=k, marker_color=bar_color,
                                 hovertemplate=str(dtime),
                                 text=[display_text], textangle=0, textfont=dict(size=text_size)))

    fig.update_traces(marker_line_color='black', marker_line_width=.8, opacity=1)

    # Customize the appearance
    if size < 20:
        height = 500
    elif size < 50:
        height = 1000
    else:
        height = 1500

    fig.update_layout(
        # autosize=False,
        height=height,
        title=f"StochRSI Map, {filtered_df['tf'].unique()[0]}",
        yaxis_title='Symbol',
        xaxis_title='Bars',
        barmode='stack',  # Stacks the bars on top of each other
        showlegend=False,  # Hide the color legend
    )
    return fig
