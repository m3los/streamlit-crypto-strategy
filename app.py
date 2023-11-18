import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import ccxt
import pandas_ta as pta
from datetime import datetime
from time import perf_counter
from functools import reduce

st.set_page_config(layout="wide", page_title="Supply & Demand", page_icon="📊")
import tradingview_charts as tv
import css
import plot
import discord

css.apply_css()
start = perf_counter()

if "autorun" not in st.session_state:
    st.session_state["autorun"] = False
if "symbols" not in st.session_state:
    st.session_state["symbols"] = []
if "last_data" not in st.session_state:
    st.session_state["last_data"] = None
if "discord_url" not in st.session_state:
    st.session_state["discord_url"] = None


# fetch symbol
def fetch_data(ex, sym, tf, sin, lmt):
    ohlcv = ex.fetch_ohlcv(sym, tf, sin, lmt)
    df = pd.DataFrame(ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df['symbol'] = sym
    df['tf'] = tf
    df = df[:-1]
    return df


# Trend definition
def trend(x):
    if (x['ema50'] > x['ema100']) and (x['ema100'] > x['ema200']) and (x['ema100'] > x['sma200']):
        return 'bull'
    elif (x['ema50'] < x['ema100']) and (x['ema100'] < x['ema200']) and (x['ema100'] < x['sma200']):
        return 'bear'
    return 'mix'


@st.cache_resource
def connect_exchange(ex: str):
    if ex == "binance":
        return ccxt.binance({"enableRateLimit": True})
    if ex == "kucoin":
        return ccxt.kucoin({"enableRateLimit": True})
    if ex == "bitget":
        return ccxt.bitget({"enableRateLimit": True})
    if ex == "okx":
        return ccxt.okx({"enableRateLimit": True})


def get_n_symbols(ex) -> list:
    if pair_limit == 0:
        return []

    usdt_symbols = [sym for sym in ex.symbols if
                    ("USDT" in sym and "USDC" not in sym and "USD/" not in sym and not sym.startswith("USDT"))]

    # filter futures(swap) symbols
    usdt_symbols = [sym for sym in usdt_symbols if ":USDT" not in sym]
    if ex.has["fetchTickers"]:
        tickers = ex.fetch_tickers(usdt_symbols)
    else:
        raise Exception(ex.id + ' does not have the fetch_tickers method')

    sym_vol = {}
    for key in tickers.keys():
        sym_vol[key] = tickers[key]["quoteVolume"]

    # sort dict by value(volume24h)
    sort = dict(sorted(sym_vol.items(), key=lambda x: x[1], reverse=True))
    # return list of n symbols
    return [*sort][:pair_limit]


# get n usdt symbols sorted by highest 24h volume
def get_custom_symbols(cust_symbols: str) -> list:
    return cust_symbols.split()


def get_all_symbols(ex, cust_symbols) -> list:
    all_symbols = get_n_symbols(ex) + get_custom_symbols(cust_symbols)
    # remove duplicates
    lst = []
    for sym in all_symbols:
        if sym not in lst:
            lst.append(sym)
    return lst


def stop_autorun():
    st.session_state["autorun"] = False


def add_indicators(df) -> pd.DataFrame:
    df['hma55'] = pta.hma(df['close'], length=55)
    df['ema20'] = pta.ema(df['close'], length=20)
    df['ema50'] = pta.ema(df['close'], length=50)
    df['ema100'] = pta.ema(df['close'], length=100)
    df['ema200'] = pta.ema(df['close'], length=200)
    df['sma200'] = pta.sma(df['close'], length=200)

    df['trend'] = df.apply(trend, axis=1)

    df['e_mean'] = df[['ema20', 'ema50', 'ema100', 'ema200']].mean(axis=1)
    df['e_std'] = df[['ema20', 'ema50', 'ema100', 'ema200']].std(axis=1,
                                                                 ddof=0)  # std population, not normalized
    STD_MULTIPLIER = 2
    df['lower_band'] = df['e_mean'] - (df['e_std'] * STD_MULTIPLIER)
    df['upper_band'] = df['e_mean'] + (df['e_std'] * STD_MULTIPLIER)

    df['q_mean'] = df[['e_mean', 'lower_band', 'upper_band', 'hma55']].mean(axis=1)
    df['q_std'] = df[['e_mean', 'lower_band', 'upper_band', 'hma55']].std(axis=1, ddof=0)
    df['q_lower_band'] = df['q_mean'] - df['q_std']
    df['q_upper_band'] = df['q_mean'] + df['q_std']

    # custom stoch rsi
    stoch_rsi = pta.stochrsi(df['q_mean'])  # returns df with columns k,d
    stoch_rsi.rename(columns={'STOCHRSIk_14_14_3_3': 'stochrsi_k', 'STOCHRSId_14_14_3_3': 'stochrsi_d'}, inplace=True)
    # if not round() 'k' displays as 0 but in reality it's some weird shit like 0.00000044515 which != 0
    stoch_rsi = stoch_rsi.round(decimals=2)
    df = pd.concat([df, stoch_rsi], axis=1)

    # conditions for stochRSI indicator
    conditions = [
        ((df['trend'] == 'bull') | (df['trend'] == 'mix')) & (
                df['stochrsi_k'].shift() > df['stochrsi_k']) & (df['stochrsi_k'] < 90) & (
                df['stochrsi_k'] != 0),
        (df['trend'] == 'bear') & (df['stochrsi_k'].shift() > df['stochrsi_k']) & (df['stochrsi_k'] != 0),
        (df['stochrsi_k'] == 0) & (df['close'] < df[['ema50', 'q_mean']].min(axis=1)),
        (df['trend'] == 'bear') & (df['close'] < df[['ema50', 'q_mean']].max(axis=1)) & (
                df['stochrsi_k'].shift() <= df['stochrsi_k']),
        (((df['stochrsi_k'].shift() < df['stochrsi_k']) & (df['stochrsi_k'] > 7)) & (
                df['stochrsi_k'] < 80)) | (
                (df['trend'] == 'bear') & (df['stochrsi_k'] > 80) & (df['stochrsi_d'] < 100)),
        df['stochrsi_k'] > 80
    ]
    results = [
        "orange",   # orange
        "orange",   # orange
        "red",      # red
        "gray",     # gray
        "lime",     # lime
        "green"     # green
    ]

    # create a new column in the df where every condition has its own result
    df["bar_color"] = np.select(conditions, results)  # set to "0" by default if no condition is met
    df.loc[df["bar_color"] == "0", "bar_color"] = "gray"  # change "0" to gray

    # order blocks v1
    # df["s_d"] = np.where((df.open >= df.close) & (df.low.shift(-2) > df.high), "D", "0")
    # df.loc[(df.open <= df.close) & (df.high.shift(-2) < df.low), ["s_d"]] = "S"

    # order blocks v2
    df["s_d"] = np.where((df.high < df.close.shift(-1)) &
                         (df.low.shift(-2) > df.high) &
                         (df.close.shift(-1) > df['close'].rolling(4).max()), "D", "0")
    df.loc[(df.low > df.close.shift(-1)) &
           (df.high.shift(-2) < df.low) &
           (df.close.shift(-1) < df['close'].rolling(4).min()), ["s_d"]] = "S"
    return df


def merge_df(df1, df2):
    df1['datetime_copy'] = df1['datetime'].copy()

    def shifter():
        _tf = df1['tf'].iloc[-1]
        if _tf == '5m':
            return -4
        elif _tf == '1h' or _tf == '4h':
            return -3
        elif _tf == '15m' or _tf == '12h':
            return -2
        else:
            return 0

    df2['hma55_x'] = df2['hma55'].shift(shifter())

    # merge higher tf to lower tf, add suffix to columns with same name
    dataframe = pd.merge_ordered(df2, df1, fill_method="ffill", on='datetime', how='left', suffixes=('_df2', None))

    dataframe.drop(dataframe[dataframe['datetime'] != dataframe['datetime_copy']].index, inplace=True)
    dataframe.reset_index(inplace=True)
    del dataframe['index']

    # Enter long conditions
    long_c = []
    c1 = dataframe['low'] > dataframe['hma55_x']
    c2 = dataframe['close'] > dataframe['q_lower_band']
    long_c.extend([c1, c2])

    dataframe.loc[
        reduce(lambda x, y: x & y, long_c),
        'trade'] = 'L'

    short_c = []
    c1 = dataframe['high'] < dataframe['hma55_x']
    short_c.extend([c1])

    dataframe.loc[
        reduce(lambda x, y: x & y, short_c),
        'trade'] = 'S'

    return dataframe.tail(50)


@st.cache_data
def update_data(last_data):
    if last_data is None:
        return {}

    ex_connector = connect_exchange(exchange)
    ex_connector.load_markets()
    st.session_state["symbols"] = get_all_symbols(ex_connector, cust_pairs)
    df_dict = {}

    for tf in ohlcv_timeframes:
        df = pd.DataFrame()
        for sym in st.session_state["symbols"]:
            try:
                ohlcv = fetch_data(ex_connector, sym, tf, None, candle_limit)
                ohlcv['tf'] = tf
                ohlcv['symbol'] = sym
                if ohlcv.shape[0] < 200:
                    continue
                ohlcv = add_indicators(ohlcv)
                df = pd.concat([df, ohlcv])
            except ccxt.BadSymbol:
                st.error(f'Bad Symbol Error: {sym}', icon="❌")
                st.stop()
        df_dict[tf] = df
    return df_dict


# == SIDEBAR ==
with st.form("sidebar_form"):
    with st.sidebar:
        st.title(":green[Data] settings ⚙")
        st.divider()
        exchange = st.selectbox("**EXCHANGE**",
                                ["binance", "kucoin", "bitget", "okx"], 0)
        ohlcv_timeframes = st.multiselect("**TIMEFRAMES**", ["1m", "5m", "15m", "1h", "4h", "12h", "1d"],
                                          default=["1h", "15m"], max_selections=3)
        pair_limit = st.slider("**PAIR LIMIT**", min_value=0, max_value=100, value=5, step=1,
                               help="Pairs sorted by highest 24h$ volume")
        candle_limit = st.slider("**CANDLE LIMIT**", min_value=10, max_value=1000, value=1000, step=10,
                                 help="Number of candles to fetch for each pair")

        cust_pairs = st.text_area("**CUSTOM PAIRS**",
                                  placeholder="BTC/USDT ETH/USDT")
        st.divider()
        submitted = st.form_submit_button("Fetch")

        # after click "Run" (or any form_submit_button), it normaly just reloads the page
        if submitted:
            now = datetime.now().isoformat(sep=' ')
            st.session_state['last_data'] = now


# == HEADER ==
st.subheader("Crypto :orange[strategy] Dashboard ⬆⬇")
st.write(f"Last data retrieved: {st.session_state['last_data']}, {exchange}")

# == INFO HEADER ==
col1, col2 = st.columns(2)
with st.container():
    with col1:
        light = "🟢" if st.session_state["autorun"] else "🔴"
        st.button(f"Stop Autorun {light}",
                  on_click=stop_autorun, key="stop_btn", disabled=not st.session_state["autorun"])
    with col2:
        st.empty()

# == DASHBOARD FORM ==
with st.form("select_form"):
    st.subheader(":green[Strategy] settings")

    col1, col2, col3 = st.columns([5, 1, 2])
    with st.container():
        with col1:
            bar_limit = st.slider("BAR LIMIT", min_value=5, max_value=50, value=30, step=1, key="bar slider",
                                  help="Show last n bars")
            timeframes = st.multiselect("**TIMEFRAME**", ohlcv_timeframes)
            plots = st.multiselect("**PLOT CHARTS**",
                                   ["StochRSI Map", "Adv StochRSI Map", "Performance Lines", "Acceleration",
                                    "Acceleration_2", "Order Blocks"],
                                   max_selections=2)
            discord_notif = st.multiselect("**SEND DISCORD NOTIFICATION**",
                                           ["StochRSI", "Adv StochRSI", "Order Blocks", "test"])
            discord_url = st.text_input("**DISCORD URL**", st.session_state["discord_url"])
        with col2:
            st.empty()
        with col3:
            arun_chkbox = st.checkbox("AUTORUN")
            autorun_interval = st.number_input("INTERVAL", min_value=1, max_value=1440,
                                               help="Rerun every n minutes")
            autorun_limit = st.number_input("LIMIT", min_value=10, max_value=1000,
                                            help="Number of reruns")
            st.divider()
            run_btn = st.form_submit_button("✨ Run ✨", use_container_width=True)
    if run_btn:
        st.session_state["autorun"] = arun_chkbox
        st.session_state["discord_url"] = discord_url
        if not st.session_state['symbols']:
            st.info("No new data")

if st.session_state["autorun"]:
    rerun_counter = st_autorefresh(interval=autorun_interval * 1000 * 60, limit=autorun_limit, key="rerun_counter")
    now = datetime.now().isoformat(sep=' ')
    st.session_state['last_data'] = now

tv_chkbox = st.checkbox("Show TV chart")

# cached data {tf1: df1, tf2: df2,}
with st.spinner("Downloading data"):
    tf_dict = update_data(st.session_state['last_data'])

st.write('Selected pairs:')
if st.session_state["symbols"]:
    st.write(
        f"""
        <style>orange {{ color: orange }} silver {{ color: silver }}</style>
        <p>
        <orange>{' '.join([sym.split('/')[0] for sym in st.session_state['symbols']])}</orange>
        <silver> /{''.join([sym for sym in st.session_state['symbols'] if sym.split('/')[1] != 'USDT'])}USDT</silver>
        </p>
        """, unsafe_allow_html=True)
else:
    st.write("<p style='font-size:16px; color:silver;'>No new data!</p>", unsafe_allow_html=True)

s1 = perf_counter()
if ("Adv StochRSI Map" in plots or "AdvStochRSI" in discord_notif) and st.session_state['symbols']:
    if len(timeframes) == 2:
        ac55_df = pd.DataFrame()

        for symbol in tf_dict[timeframes[0]]['symbol'].unique():
            df_1 = tf_dict[timeframes[0]].loc[tf_dict[timeframes[0]]['symbol'] == symbol]
            df_2 = tf_dict[timeframes[1]].loc[tf_dict[timeframes[1]]['symbol'] == symbol]
            df_merged = merge_df(df_1, df_2)
            ac55_df = pd.concat([ac55_df, df_merged])

        # Filter only last n rows for each symbol
        filtered_df = ac55_df.groupby('symbol').tail(bar_limit)

        if "Adv StochRSI Map" in plots:
            st.plotly_chart(plot.plot_adv_map(filtered_df, st.session_state['last_data']))

        # Discord notification
        if "Adv StochRSI" in discord_notif:
            fields = {}
            for symbol in filtered_df['symbol'].unique():
                symbol_df = filtered_df.loc[filtered_df['symbol'] == symbol]

                last_bar = symbol_df.iloc[-1]
                secLast_bar = symbol_df.iloc[-2]

                bull = (last_bar['trend'] == 'bull') & (last_bar['trade'] == 'L') & (last_bar['bar_color'] == 'gray')
                mix = (last_bar['trend'] == 'mix') & (last_bar['trade'] == 'L') & \
                      ((last_bar['bar_color'] == 'gray') | (last_bar['bar_color'] == 'lime'))
                bear = (last_bar['trend'] == 'bear') & (last_bar['trade'] == 'L') & \
                       ((last_bar['bar_color'] == 'gray') | (last_bar['bar_color'] == 'lime'))

                if bull or mix or bear:
                    fields[symbol] = f"({last_bar['trend']} \n {secLast_bar['bar_color']} -> {last_bar['bar_color']})"
            if fields:
                discord.send_msg("Bar color change", timeframes[0], fields, discord_url)
    else:
        st.error("2 timeframes are needed to plot 'AdvStochRSI Map'", icon="🚨")

if (plots or discord_notif) and st.session_state['symbols']:
    # Plot only one tf if StochRSI Map chart is selected
    _range = len(timeframes) if "Adv StochRSI Map" not in plots else 1

    for t, timeframe in enumerate(timeframes):
        if t == _range:
            break
        tf_df = tf_dict[timeframe]
        filtered_data = tf_df.groupby('symbol').tail(bar_limit)

        # Discord notifications
        if "Order Blocks" in discord_notif:
            fields = {}
            for symbol in filtered_data['symbol'].unique():
                if filtered_data.iloc[-3]['s_d'] != "0":
                    fields[symbol] = f"New {filtered_data.iloc[-3]['s_d']} zone"
            if fields:
                discord.send_msg("Order Blocks", timeframe, fields, discord_url)

        if "StochRSI" in discord_notif:
            fields = {}
            for symbol in filtered_data['symbol'].unique():
                symbol_df = filtered_data.loc[filtered_data['symbol'] == symbol]

                last_bar_color = symbol_df.iloc[-1]['bar_color']
                secLast_bar_color = symbol_df.iloc[-2]['bar_color']

                signal = ((last_bar_color == "lime") & (secLast_bar_color == "gray")) | (
                        (last_bar_color == "lime") & (secLast_bar_color == "orange")) | (
                                 (last_bar_color == "orange") & (secLast_bar_color == "green"))
                if signal:
                    fields[symbol] = f"({secLast_bar_color} -> {last_bar_color})"
            if fields:
                discord.send_msg("Bar color change", timeframe, fields, discord_url)  # send discord msg

        if "test" in discord_notif:
            discord.send_msg("test", timeframe, {"symbol": "val"}, discord_url)  # send discord msg

        # Plot
        if "StochRSI Map" in plots:
            st.plotly_chart(plot.plot_map(filtered_data), True)
        if "Performance Lines" in plots:
            st.plotly_chart(plot.plot_spaghetti(filtered_data), True)
        if "Acceleration" in plots:
            st.plotly_chart(plot.plot_acceleration(filtered_data), True)
        if "Acceleration_2" in plots:
            st.plotly_chart(plot.plot_acceleration2(filtered_data), True)
        if "Order Blocks" in plots:
            st.plotly_chart(plot.plot_order_blocks(filtered_data), True)
    st.caption(f"Plot execution time: {perf_counter() - s1} sec")

st.divider()

# == TV CHART ==
if tv_chkbox:
    default_tf = "1h" if not timeframes else timeframes[0]
    components.html(tv.chart_widget(exchange, st.session_state['symbols'], default_tf), height=800)

st.caption(f"Execution time: {perf_counter() - start} sec")
