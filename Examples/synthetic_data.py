# https://pub.towardsai.net/procedural-ochl-stock-generator-54ce041931be

import plotly.express as px

# import pandas_ta as ta
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
from scipy.stats import skewnorm


# graphs all columns for x, uses df.index as y
def graph_stock(df, title):
    # time_column is a string

    pd.options.plotting.backend = "plotly"
    # fig = px.line(df, x='time', y=['price', 'price_2'])
    fig = px.line(df, x=df.index, y=df.columns, title=title)
    fig.show()


def graph_OHLC(df, title=""):
    # fig_1 = px.line(df, x=df.index, y=df.columns, title=title)
    fig_2 = go.Figure(
        data=go.Candlestick(
            x=df.index,
            title=title,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        )
    )
    fig_2.update(layout_xaxis_rangeslider_visible=False)
    fig_2.show()


def simulate_stock(initial_price, drift, volatility, trend, days):
    """
    It is now time to create the main function:
    a code that can perform the entire simulation when called.
    I will use this function using 8760 as a length parameter,
    meaning that I will be generating random values for every hour of the year.
    Because the results can easily vary from the original 1000 as
    base price (after 1 year I may end up having 100,000, not very realistic),
     I will drastically diminish my random distribution values, approximating them to 0.
    """

    def create_pdf(sd, mean, alfa):
        # invertire il segno di alfa
        x = skewnorm.rvs(alfa, size=1000000)

        def calc(k, sd, mean):
            return (k * sd) + mean

        x = calc(x, sd, mean)  # standard distribution

        # graph pdf
        # pd.DataFrame(x).hist(bins=100)

        # pick one random number from the distribution
        # formally I would use cdf, but I just have to pick randomly from the 1000000 samples
        # np.random.choice(x)
        return x

    def create_empty_df(days):
        # creare un empty DataFrame con le date
        empty = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=days, freq="D"))
        empty = pd.DataFrame(empty)
        # si tagliano ore, minuti, secondi
        empty

        # si tagliano ore, minuti, secondi
        empty.index = [str(x)[0 : empty.shape[0]] for x in list(empty.pop(0))]
        empty

        # final dataset con values
        stock = pd.DataFrame([x for x in range(0, empty.shape[0])])
        stock.index = empty.index
        return stock

    # skeleton
    stock = create_empty_df(days)

    # initial price
    stock[0][0] = initial_price

    # create entire stock DataFrame
    x = create_pdf(volatility, drift, trend)
    for _ in range(1, stock.shape[0]):
        stock.iloc[_] = stock.iloc[_ - 1] * (1 + np.random.choice(x))

    stock.index = pd.DatetimeIndex(stock.index)
    return stock

    # normal distribution for difference between low and high


def create_pdf(sd, mean, alfa):
    # invertire il segno di alfa
    x = skewnorm.rvs(alfa, size=1000000)

    def calc(k, sd, mean):
        return (k * sd) + mean

    x = calc(x, sd, mean)  # standard distribution

    # graph pdf
    # pd.DataFrame(x).hist(bins=100)

    # pick one random number from the distribution
    # formally I would use cdf, but I just have to pick randomly from the 1000000 samples
    # np.random.choice(x)
    return x


def OHLC(group_values):
    """
    I will be using the OCHL converter to extract high, low, open, and close values
    of a tranche of a 24-hour stock price.
    """
    min_ = min(group_values)
    max_ = max(group_values)
    range = max_ - min_
    open = min_ + range * random.random()
    close = min_ + range * random.random()
    return min_, max_, open, close


def main() -> None:
    ran = create_pdf(0.1, 0.2, 0)
    np.random.choice(ran)

    df = simulate_stock(
        initial_price=1000, drift=0, volatility=0.01, trend=0, days=8760
    )

    # Once I have simulated one year of data, I can finally extract the OCHL.
    # To do this, I will group the data by tranches of 24, then use the OCHL converter.
    df_ = list()

    # df.groupby(np.arange(len(df))//24).apply(OCHL) non funziona
    # sarebbe il modo corretto, ma devo creare un nuovo df da 0
    for a, b in df.groupby(np.arange(len(df)) // 24):
        group_values = np.array(b.values).flatten()
        low, high, open, close = OHLC(group_values)
        df_.append([low, high, open, close])

    df_OHLC = pd.DataFrame(
        df_,
        index=pd.Series(pd.date_range("2020-01-01", periods=365, freq="D")),
        columns=["low", "high", "open", "close"],
    )

    # graph_stock(df, "")

    fig = go.Figure(
        data=go.Candlestick(
            x=df_OHLC.index,
            open=df_OHLC["open"],
            high=df_OHLC["high"],
            low=df_OHLC["low"],
            close=df_OHLC["close"],
        )
    )
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()


if __name__ == "__main__":
    main()
