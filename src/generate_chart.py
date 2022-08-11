import random
import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants.constants import *

"""
Praews notes: 
    The bodies of the fake candles look very similar to each other
        1. There are many candles of the same body size
        2. There are also many candle wicks of the same size

"""

data_date_ranges = {
    # spot
    BINANCE_BTCUSDT_DAY: {"start_date": "2019-09-08", "end_date": "2022-06-16"},
    BINANCE_AAVEUSDT_DAY: {"start_date": "2020-10-16", "end_date": "2022-06-16"},
    BINANCE_ADAUSDT_DAY: {"start_date": "2018-04-17", "end_date": "2022-07-30"},
    BINANCE_CELRUSDT_DAY: {"start_date": "2019-03-25", "end_date": "2022-07-30"},
    BINANCE_DASHUSDT_DAY: {"start_date": "2019-03-28", "end_date": "2022-07-30"},
    BINANCE_DOGEUSDT_DAY: {"start_date": "2020-07-10", "end_date": "2022-07-30"},
    BINANCE_DOTUSDT_DAY: {"start_date": "2020-08-18", "end_date": "2022-07-30"},
    BINANCE_ETCUSDT_DAY: {"start_date": "2018-06-12", "end_date": "2022-07-30"},
    BINANCE_ETHUSDT_DAY: {"start_date": "2017-08-17", "end_date": "2022-07-30"},
    # spot
    BINANCE_ETHUSDT_FUTURES_DAY: {"start_date": "2019-11-27", "end_date": "2022-03-15"},
    BINANCE_LTCUSDT_FUTURES_DAY: {"start_date": "2020-01-09", "end_date": "2022-03-15"},
    BINANCE_ADAUSDT_FUTURES_DAY: {"start_date": "2020-01-31", "end_date": "2022-07-30"},
    BINANCE_BTCUSDT_FUTURES_DAY: {"start_date": "2019-09-08", "end_date": "2022-03-15"},
    BINANCE_XMRUSDT_FUTURES_DAY: {"start_date": "2020-02-03", "end_date": "2022-07-30"},
}


def get_config() -> dict:
    return {
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "eraseshape",
        ],
        "scrollZoom": True,
        "doubleClickDelay": 1000,  # double click the graph to reset position
        "displayModeBar": True,
    }


def brownian_motion(rows: int) -> np.ndarray:
    """Creates a brownian motion ndarray"""
    T = 1.0
    dimensions = 1
    times = np.linspace(0.0, T, rows)
    dt = times[1] - times[0]

    # Bt2 - Bt1 ~ Normal with mean 0 and variance t2-t1
    dB = np.sqrt(dt) * np.random.normal(size=(rows - 1, dimensions))
    B0 = np.zeros(shape=(1, dimensions))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
    return B


def brownian_motion_distribution(rows: int = 2880) -> list[float]:
    """Returns a dataframe with a brownian motion distribution"""
    bm_array = brownian_motion(rows)
    bm_array = [i[0] for i in bm_array]
    return normalize_ohlc_list(bm_array)


def normalize_ohlc_list(data: list) -> list:
    """Normalize OHLC data with random multiplier
    normalization formula: (data - min) / (max - min)
    """
    random_multiplier = random.randint(9, 999)

    _max = np.max(data)
    _min = np.min(data)

    norm_data: np.ndarray = ((data - _min) / (_max - _min)).tolist()
    return [round(i * random_multiplier, 4) for i in norm_data]


def normalize_ohlc_data(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
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

    o = round(norm_open * random_multiplier, 4)
    h = round(norm_high * random_multiplier, 4)
    l = round(norm_low * random_multiplier, 4)
    c = round(norm_close * random_multiplier, 4)
    return o, h, l, c


def generate_random_crypto_df(
    days: int, start_price: float, col_name: str, start_date: str, volatility: int
) -> pd.DataFrame:
    """Need to generate 1min, 5min, 15min, 1hr, 4hr, 1day, 1week"""

    distribution_dict = {
        1: np.random.normal,
        2: np.random.laplace,
        3: np.random.logistic,
    }

    # periods = days * 1440 # minutes
    periods = days * 24  # hours
    # periods = days        # days
    # periods = days / 7    # weeks ?

    # randomly pick a distribution function
    dist_func = distribution_dict.get(random.randint(1, 3))
    steps = dist_func(loc=0, scale=volatility, size=periods)
    steps[0] = 0

    price = start_price + np.cumsum(steps)
    price = [round(abs(p), 6) for p in price]

    return pd.DataFrame(
        {
            "ticker": np.repeat([col_name], periods),
            # <------ TEST THIS WITH 'D' INSTEAD OF 'H' <------
            "date": np.tile(pd.date_range(start_date, periods=periods, freq="H"), 1),
            "price": (price),
        }
    )


def downsample_ohlc_data(df: pd.DataFrame, timeframe: str) -> None:
    """Converts a higher resolution dataframe into a lower one.
    For example: converts 1min candle sticks into 5min candle sticks.

    # A helpful hint
    # https://towardsdatascience.com/pandas-resample-tricks-you-should-know-for-manipulating-time-series-data-7e9643a7e7f3

    TODO:
        Resample time frame to 1min, 5min, 15min, 1hr, 4hr, 1d, 1w
    """

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # print(df.index[:7][-1], df['open'][0], max(df['high'][:7]), min(df['low'][:7]), df['close'][:7][-1])

    df = df.resample("W").aggregate(
        {
            "open": lambda s: s[0],
            "high": lambda df: df.max(),
            "low": lambda df: df.min(),
            "close": lambda df: df[-1],
        }
    )
    return df


def create_dates(
    num_days_range: int, start_date_limit: str, end_date_limit: str
) -> tuple[str, str]:
    """Randomly pick a start and end date within the given starting and ending bounds"""

    start_date_limit_l = [int(i) for i in start_date_limit.split("-")]
    end_date_limit_l = [int(i) for i in end_date_limit.split("-")]

    start_limit_dt = datetime.datetime(
        year=start_date_limit_l[0],
        month=start_date_limit_l[1],
        day=start_date_limit_l[2],
    )

    end_limit_dt = datetime.datetime(
        year=end_date_limit_l[0], month=end_date_limit_l[1], day=end_date_limit_l[2]
    )

    # get the number of days from the start date to the end date
    date_range_limit = end_limit_dt - start_limit_dt

    # create a list of all the dates within the given date bounds
    dt_list = [
        start_limit_dt + datetime.timedelta(days=x)
        for x in range(date_range_limit.days)
    ]

    # pick a random day to start minus the given range
    start_i = random.randint(0, len(dt_list) - num_days_range)
    end_i = start_i + num_days_range

    start_random_dt = dt_list[start_i]
    end_random_dt = dt_list[end_i]
    return start_random_dt.strftime("%Y-%m-%d"), end_random_dt.strftime("%Y-%m-%d")


def real_case(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Create a dataframe for real data"""
    df = df.drop(df[df["date"] < start_date].index)
    df = df.drop(df[df["date"] > end_date].index)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df.set_index("date", inplace=True)
    return df


def random_case(
    num_days: int, start_price: float, asset_name: str, start_date: str, volatility: int
) -> pd.DataFrame:
    """Create a dataframe for random data"""

    df = generate_random_crypto_df(
        num_days, start_price, asset_name, start_date, volatility=volatility
    )
    df.index = pd.to_datetime(df.date)

    # 25% to use a brownian motion distribution instead
    if random.randint(1, 4) == 4:
        df["price"] = brownian_motion_distribution()

    # https://stackoverflow.com/questions/17001389/pandas-resample-documentation
    return df.price.resample("D").ohlc()


def create_half_df(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns new series objects that contain half the rows as the input series objects"""
    return (
        open.iloc[: len(open) // 2],
        high.iloc[: len(high) // 2],
        low.iloc[: len(low) // 2],
        close.iloc[: len(close) // 2],
    )


def create_whale_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a modified df containing whale values.
    Iterates over the dataframe and extends all values
    of the candle given a random chance.

    A single graph will have a random chance until a new graph is created.
    If the random number chosen is less than or equal to random_chance, create a whale candle.

    whale_mult:
        assigns a random floating point multiplier between the range of [WHALE_LOWER_MULT, WHALE_UPPER_MULT].
        By doing this, every time a whale candle is created, the probability
        of it being stretched with the same ratio as the
        previous whale candle or even the next whale candle is essentially 0%
    """

    # probability for creating a whale candle will be from 1-20%
    random_chance = random.randint(1, 20)

    df.reset_index(inplace=True)

    for i in range(len(df)):
        if random.randint(1, 100) <= random_chance:
            # assigns a random floating point multiplier between the range of [WHALE_LOWER_MULT, WHALE_UPPER_MULT].
            # By doing this, every time a whale candle is created, the probability
            # of it being stretched with the same ratio as the
            # previous whale candle or even the next whale candle is essentially 0%
            whale_mult = random.uniform(WHALE_LOWER_MULT, WHALE_UPPER_MULT)

            df.at[i, "open"] = df.iloc[i]["open"] * whale_mult
            df.at[i, "high"] = df.iloc[i]["high"] * whale_mult
            df.at[i, "low"] = df.iloc[i]["low"] / whale_mult
            df.at[i, "close"] = df.iloc[i]["close"] / whale_mult

    df.set_index("date", inplace=True)
    return df


def extend_wicks(df: pd.DataFrame, hl_mult: float) -> pd.DataFrame:
    """Returns a dataframe with the high and low wicks multiplied by the passed in hl_mult"""
    df.reset_index(inplace=True)
    for i in range(len(df)):
        new_h = df.iloc[i]["high"] * hl_mult
        new_l = df.iloc[i]["low"] - (df.iloc[i]["low"] * (hl_mult - 1))

        df.at[i, "high"] = new_h
        df.at[i, "low"] = new_l
    df.set_index("date", inplace=True)
    return df


def extend_all_wicks_randomly(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with the highs and lows multiplied by a random float"""
    df.reset_index(inplace=True)
    for i in range(len(df)):
        h_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)
        l_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)

        new_h = df.iloc[i]["high"] * h_mult
        new_l = df.iloc[i]["low"] - (df.iloc[i]["low"] * (l_mult - 1))

        df.at[i, "high"] = new_h
        df.at[i, "low"] = new_l
    df.set_index("date", inplace=True)

    df = extend_wicks_randomly(df)
    return df


def extend_wicks_randomly(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with the highs, lows multiplied by a random float

    3 possibilities:
        extend only the high
        extend only the low
        extend both
    """

    df.reset_index(inplace=True)
    for i in range(len(df)):
        h_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)
        l_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)

        random_choice = random.randint(1, 3)

        if random_choice == 1:
            # extend only the high
            df.at[i, "high"] = df.iloc[i]["high"] * h_mult
        elif random_choice == 2:
            # extend only the low
            df.at[i, "low"] = df.iloc[i]["low"] - (df.iloc[i]["low"] * (l_mult - 1))
        else:
            # extend both
            df.at[i, "high"] = df.iloc[i]["high"] * h_mult
            df.at[i, "low"] = df.iloc[i]["low"] - (df.iloc[i]["low"] * (l_mult - 1))
    df.set_index("date", inplace=True)
    return df


def connect_open_close_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe where every candles close is the next candles open.
    This is needed because cryptocurrencies run 24/7.
    There are no breaks or pauses so each candle is connected to the next candle.
    """
    df.reset_index(inplace=True)

    for i in range(1, len(df)):
        # connects each open and close together
        df.at[i, "open"] = df.iloc[i - 1]["close"]

        min_value = min(df.iloc[i]["open"], df.iloc[i]["high"], df.iloc[i]["close"])

        max_value = max(df.iloc[i]["open"], df.iloc[i]["low"], df.iloc[i]["close"])

        # something went wrong and the low is not the lowest value
        if df.iloc[i]["low"] > min_value:
            # get the difference between the low and the lowest value and subtract it from the low
            df.at[i, "low"] = df.iloc[i]["low"] - abs(min_value - df.iloc[i]["low"])

        # get the difference between the highest value and the high and add it to the high
        if df.iloc[i]["high"] < max_value:
            df.at[i, "high"] = df.iloc[i]["high"] + abs(max_value - df.iloc[i]["high"])

    df.set_index("date", inplace=True)
    return df


def create_figure(
    index: pd.RangeIndex,
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    answer=None,
) -> go.Figure:
    fig = go.Figure(
        data=go.Candlestick(
            x=index,
            open=open,
            high=high,
            low=low,
            close=close,
        )
    )

    # fig.update_yaxes(showticklabels=False)
    # fig.update_xaxes(showticklabels=False)

    fig.update_layout(
        template="plotly_dark",
        title=answer,
        xaxis_title="Date",
        yaxis_title="Price",
        dragmode="zoom",
        newshape_line_color="white",
        font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
        # hides the xaxis range slider
        xaxis=dict(rangeslider=dict(visible=False)),
    )
    return fig


def main() -> None:
    start_price = 100_000
    asset_name = "Unknown"
    total_graphs = 1
    num_days_range = 120
    answers = {}
    days = 91

    for i in range(total_graphs):
        # pick a data set randomly
        data_choice = random.choice(list(data_date_ranges.keys()))

        d_range = data_date_ranges.get(data_choice)
        start_date__limit_l = [int(i) for i in d_range["start_date"].split("-")]

        # adjust the start_date 91 days after the original start date
        start_dt_limit = datetime.datetime(
            year=start_date__limit_l[0],
            month=start_date__limit_l[1],
            day=start_date__limit_l[2],
        )
        adjusted_start_dt = start_dt_limit + datetime.timedelta(days=days)
        adj_start_date_limit = adjusted_start_dt.strftime("%Y-%m-%d")
        end_date_limit = d_range["end_date"]

        df_real = pd.read_csv(
            data_choice,
            usecols=["date", "symbol", "open", "high", "low", "close"],
            skiprows=1,
        )

        # reverse the data
        df_real = df_real[::-1]

        start_date, end_date = create_dates(
            num_days_range, adj_start_date_limit, end_date_limit
        )

        df = None

        # if number is 1 generate real df, else: generate fake (aka random)
        # if random.randint(0, 1):
        if False:
            df = df_real.copy()
            df = real_case(df, start_date, end_date)
            answers[i] = f"Real: {start_date} to {end_date} {data_choice}"
        else:
            volatility = random.randint(100, 200)
            df = random_case(
                num_days_range, start_price, asset_name, start_date, volatility
            )
            answers[i] = f"Fake: {start_date} to {end_date}"

            # make the candles look a little bit more real
            #############################################
            df = create_whale_candles(df)
            df = extend_all_wicks_randomly(df)
            df = connect_open_close_candles(df)
            #############################################

        norm_open, norm_high, norm_low, norm_close = normalize_ohlc_data(
            df["open"], df["high"], df["low"], df["close"]
        )

        df.reset_index(inplace=True)

        # testdf = downsample_ohlc_data(df, '4h')

        df.drop(columns=["date"], inplace=True)

        # create a new df that contains only half the dates and prices
        half_df = df.iloc[: len(df) // 2]

        half_norm_open, half_norm_high, half_norm_low, half_norm_close = create_half_df(
            norm_open, norm_high, norm_low, norm_close
        )

        fig = create_figure(
            half_df.index,
            half_norm_open,
            half_norm_high,
            half_norm_low,
            half_norm_close,
            f"HABC/USD #{i}",
        )
        fig.write_html(
            file=f"html/HABC-USD_{i}.html",
            config=get_config(),
            include_plotlyjs=True,
            validate=True,
        )
        fig.show(config=get_config())

        fig = create_figure(
            df.index, norm_open, norm_high, norm_low, norm_close, f"FABC/USD {i}"
        )
        fig.write_html(
            file=f"html/FABC-USD_{i}.html",
            config=get_config(),
            include_plotlyjs=True,
            validate=True,
        )
        fig.show(config=get_config())


if __name__ == "__main__":
    main()
