import random
import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants.constants import *


def normalize_ohlc_data(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
                        ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Normalize OHLC data with random multiplier
        normalization formula: (data - min) / (max - min)
    """

    _max = np.max([np.max(open), np.max(high), np.max(low), np.max(close)])
    _min = np.min([np.min(open), np.min(high), np.min(low), np.min(close)])

    norm_open = (open - _min) / (_max - _min)
    norm_high = (high - _min) / (_max - _min)
    norm_low = (low - _min) / (_max - _min)
    norm_close = (close - _min) / (_max - _min)

    random_multiplier = random.randint(9, 999)

    o = round(norm_open*random_multiplier, 4)
    h = round(norm_high*random_multiplier, 4)
    l = round(norm_low*random_multiplier, 4)
    c = round(norm_close*random_multiplier, 4)
    return o, h, l, c


def generate_random_crypto_df(days: int, start_price: float,
                              col_name: str, start_date: str, volatility: int) -> pd.DataFrame:
    """Need to generate 1min, 5min, 15min, 1hr, 4hr, 1day, 1week"""

    distribution_dict = {
        1: np.random.normal,
        2: np.random.laplace,
        3: np.random.logistic,
    }

    # periods = days * 1440 # minutes
    periods = days * 24     # hours
    # periods = days        # days
    # periods = days / 7    # weeks ?

    # randomly pick a distribution function
    dist_func = distribution_dict.get(random.randint(1, 3))
    steps = dist_func(loc=0, scale=volatility, size=periods)
    steps[0] = 0

    price = start_price + np.cumsum(steps)
    price = [round(abs(p), 6) for p in price]
    # price = [round(p, 6) for p in price]

    return pd.DataFrame({
        'ticker': np.repeat([col_name], periods),
        'date': np.tile(pd.date_range(start_date, periods=periods, freq='H'), 1),
        'price': (price)})


def downsample_ohlc_data(df: pd.DataFrame, timeframe: str) -> None:
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # print(df.index[:7][-1], df['open'][0], max(df['high'][:7]), min(df['low'][:7]), df['close'][:7][-1])

    df = df.resample('W').aggregate({
        'open': lambda s: s[0],
        'high': lambda df: df.max(),
        'low': lambda df: df.min(),
        'close': lambda df: df[-1]
    })

    return df


def create_dates(num_days_range: int, start_date_limit: str, end_date_limit: str) -> tuple[str, str]:
    """Randomly pick a start and end date within the given starting and ending bounds"""

    start_date_limit_l = [int(i) for i in start_date_limit.split('-')]
    end_date_limit_l = [int(i) for i in end_date_limit.split('-')]

    start_limit_dt = datetime.datetime(year=start_date_limit_l[0],
                                       month=start_date_limit_l[1],
                                       day=start_date_limit_l[2])

    end_limit_dt = datetime.datetime(year=end_date_limit_l[0],
                                     month=end_date_limit_l[1],
                                     day=end_date_limit_l[2])

    # get the number of days from the start date to the end date
    date_range_limit = end_limit_dt - start_limit_dt

    # create a list of all the dates within the given date bounds
    dt_list = [start_limit_dt +
               datetime.timedelta(days=x) for x in range(date_range_limit.days)]

    # pick a random day to start minus the given range
    start_i = random.randint(0, len(dt_list) - num_days_range)
    end_i = start_i + num_days_range

    start_random_dt = dt_list[start_i]
    end_random_dt = dt_list[end_i]
    return start_random_dt.strftime("%Y-%m-%d"), end_random_dt.strftime("%Y-%m-%d")


def real_case(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Create a dataframe for real data"""
    df = df.drop(df[df['date'] < start_date].index)
    df = df.drop(df[df['date'] > end_date].index)

    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df.set_index('date', inplace=True)
    return df


def random_case(num_days: int, start_price: float, asset_name: str,
                start_date: str, volatility: int, df: pd.DataFrame) -> pd.DataFrame:
    """Create a dataframe for random data"""
    df = generate_random_crypto_df(
        num_days, start_price, asset_name, start_date, volatility=volatility)
    df.index = pd.to_datetime(df.date)
    return df.price.resample('D').ohlc()  # how does this create ohlc?


def create_half_df(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
                   ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns new series objects that contain half the rows as the input series objects"""
    return open.iloc[:len(open)//2], high.iloc[:len(high)//2], \
        low.iloc[:len(low)//2], close.iloc[:len(close)//2]


def create_whale_candle(df: pd.DataFrame) -> pd.DataFrame:
    return df


def create_shrimp_candle(df: pd.DataFrame) -> pd.DataFrame:
    return df


def extend_wicks(df: pd.DataFrame, hl_mult: float) -> pd.DataFrame:
    """Returns a dataframe with the high and low wicks multiplied by the passed in hl_mult"""
    df.reset_index(inplace=True)
    for i in range(len(df)):
        new_h = df.iloc[i]['high'] * hl_mult
        new_l = df.iloc[i]['low'] - (df.iloc[i]['low'] * (hl_mult-1))

        df.at[i, 'high'] = new_h
        df.at[i, 'low'] = new_l
    df.set_index('date', inplace=True)
    return df


def extend_wicks_randomly(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with the highs and lows multiplied by a random float"""
    df.reset_index(inplace=True)
    for i in range(len(df)):
        h_mult = random.uniform(1, 1.001)
        l_mult = random.uniform(1, 1.001)

        new_h = df.iloc[i]['high'] * h_mult
        new_l = df.iloc[i]['low'] - (df.iloc[i]['low'] * (l_mult-1))

        df.at[i, 'high'] = new_h
        df.at[i, 'low'] = new_l
    df.set_index('date', inplace=True)
    return df


def create_figure(index: pd.RangeIndex, open: pd.Series, high: pd.Series,
                  low: pd.Series, close: pd.Series, answer=None) -> go.Figure:
    fig = go.Figure(data=go.Candlestick(
        x=index,
        open=open,
        high=high,
        low=low,
        close=close,
    ))

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)

    fig.update_layout(
        title=answer,
        xaxis_title="Time",
        yaxis_title="Value",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    return fig


def main() -> None:
    start_price = 100_000
    asset_name = 'Unknown'
    total_graphs = 1
    num_days_range = 120
    answers = {}

    date_range_dict = {
        BINANCE_BTCUSDT_DAY: {'start_date': '2019-09-08', 'end_date': '2022-06-16'},
        BINANCE_AAVEUSDT_DAY: {'start_date': '2020-10-16', 'end_date': '2022-06-16'},
        BINANCE_ADAUSDT_DAY: {'start_date': '2018-04-17', 'end_date': '2022-07-30'},
        BINANCE_CELRUSDT_DAY: {'start_date': '2019-03-25', 'end_date': '2022-07-30'},
        BINANCE_DASHUSDT_DAY: {'start_date': '2019-03-28', 'end_date': '2022-07-30'},
        BINANCE_DOGEUSDT_DAY: {'start_date': '2020-07-10', 'end_date': '2022-07-30'},
        BINANCE_DOTUSDT_DAY: {'start_date': '2020-08-18', 'end_date': '2022-07-30'},
        BINANCE_ETCUSDT_DAY: {'start_date': '2018-06-12', 'end_date': '2022-07-30'},
        BINANCE_ETHUSDT_DAY: {'start_date': '2017-08-17', 'end_date': '2022-07-30'},

        BINANCE_ETHUSDT_FUTURES_DAY: {'start_date': '2019-11-27', 'end_date': '2022-03-15'},
        BINANCE_LTCUSDT_FUTURES_DAY: {'start_date': '2020-01-09', 'end_date': '2022-03-15'},
        BINANCE_ADAUSDT_FUTURES_DAY: {'start_date': '2020-01-31', 'end_date': '2022-07-30'},
        BINANCE_BTCUSDT_FUTURES_DAY: {'start_date': '2019-09-08', 'end_date': '2022-03-15'},
        BINANCE_XMRUSDT_FUTURES_DAY: {
            'start_date': '2020-02-03', 'end_date': '2022-07-30'}
    }

    for i in range(total_graphs):
        # pick a data set randomly
        data_choice = random.choice(list(date_range_dict.keys()))

        d_range = date_range_dict.get(data_choice)
        start_date__limit_l = [int(i)
                               for i in d_range['start_date'].split('-')]

        # adjust the start_date 91 days after the original start date
        start_dt_limit = datetime.datetime(
            year=start_date__limit_l[0], month=start_date__limit_l[1], day=start_date__limit_l[2])
        adjusted_start_dt = start_dt_limit + datetime.timedelta(days=91)
        adj_start_date_limit = adjusted_start_dt.strftime("%Y-%m-%d")
        end_date_limit = d_range['end_date']

        df_real = pd.read_csv(data_choice, usecols=[
            'date', 'symbol', 'open', 'high', 'low', 'close'], skiprows=1)

        # reverse the data
        df_real = df_real[::-1]

        start_date, end_date = create_dates(
            num_days_range, adj_start_date_limit, end_date_limit)

        df = None

        # if number is 1 generate real df, else: generate fake (aka random)
        # if random.randint(0, 1):
        if True:
            # print('Real')
            df = df_real.copy()
            df = real_case(df, start_date, end_date)
            answers[i] = f"Real: {start_date} to {end_date} {data_choice}"
        else:
            # print('Fake')
            volatility = random.randint(100, 400)
            df = random_case(num_days_range, start_price, asset_name,
                             start_date, volatility, df)
            answers[i] = f"Fake: {start_date} to {end_date}"

            # df = extend_wicks(df, 1.002)
            df = extend_wicks_randomly(df)

        norm_open, norm_high, norm_low, norm_close = normalize_ohlc_data(
            df['open'], df['high'], df['low'], df['close'])

        df.reset_index(inplace=True)

        # testdf = downsample_ohlc_data(df, '4h')

        df.drop(columns=['date'], inplace=True)

        # create a new df that contains only half the dates and prices
        half_df = df.iloc[:len(df)//2]

        half_norm_open, half_norm_high, half_norm_low, half_norm_close = create_half_df(
            norm_open, norm_high, norm_low, norm_close)

        fig = create_figure(half_df.index, half_norm_open, half_norm_high,
                            half_norm_low, half_norm_close, f'H_Graph {i}')
        fig.write_html(f"html/H_Graph {i}.html")
        fig.show()

        fig = create_figure(df.index, norm_open, norm_high,
                            norm_low, norm_close, f'F_Graph {i}')
        fig.write_html(f"html/F_Graph {i}.html")
        fig.show()

        print(answers[i])

    pprint(answers)


"""
TODO:
    Resample time frame to 1min, 5min, 15min, 1hr, 4hr, 1d, 1w

"""

if __name__ == '__main__':
    main()
