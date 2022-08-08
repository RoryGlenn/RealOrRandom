from logging import getLogger
import random
import numpy as np
import pandas as pd
from brownian import brownian

from constants.constants import *

"""
TODO:
    create functions to perform the operations on the df inside the class

"""


class RandomOHLC:
    def __init__(
        self,
        num_days_range: int,
        start_price: float,
        name: str,
        volatility: int,
    ) -> None:

        self.num_days_range = num_days_range
        self.start_price = start_price
        self.name = name
        self.volatility = volatility

        self.distribution_functions = {
            1: np.random.normal,
            2: np.random.laplace,
            3: np.random.logistic,
            # 4: brownian,
        }

        self.__df: pd.DataFrame = None

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    def __brownian_motion(self, rows: int) -> np.ndarray:
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

    def __brownian_motion_distribution(self, rows: int = 2880) -> list[float]:
        """Returns a dataframe with a brownian motion distribution"""
        bm_array = self.__brownian_motion(rows)
        bm_array = [i[0] for i in bm_array]
        return self.__normalize_ohlc_list(bm_array)

    def normalize_ohlc_data(self) -> None:
        """Normalize OHLC data with random multiplier
        normalization formula: (data - min) / (max - min)
        """

        _max = np.max(
            [
                np.max(self.__df.open),
                np.max(self.__df.high),
                np.max(self.__df.low),
                np.max(self.__df.close),
            ]
        )
        _min = np.min(
            [
                np.min(self.__df.open),
                np.min(self.__df.high),
                np.min(self.__df.low),
                np.min(self.__df.close),
            ]
        )

        norm_open = (self.__df.open - _min) / (_max - _min)
        norm_high = (self.__df.high - _min) / (_max - _min)
        norm_low = (self.__df.low - _min) / (_max - _min)
        norm_close = (self.__df.close - _min) / (_max - _min)

        random_multiplier = random.randint(9, 999)

        self.__df["open"] = round(norm_open * random_multiplier, 4)
        self.__df["high"] = round(norm_high * random_multiplier, 4)
        self.__df["low"] = round(norm_low * random_multiplier, 4)
        self.__df["close"] = round(norm_close * random_multiplier, 4)
        return self.__df

    def __normalize_ohlc_list(self, data: list) -> list:
        """Normalize OHLC data with random multiplier
        normalization formula: (data - min) / (max - min)
        """
        random_multiplier = random.randint(9, 999)

        _max = np.max(data)
        _min = np.min(data)

        norm_data: np.ndarray = ((data - _min) / (_max - _min)).tolist()
        return [round(i * random_multiplier, 4) for i in norm_data]

    def __generate_random_df(
        self,
        days: int,
        start_price: float,
        col_name: str,
        volatility: int,
    ) -> None:
        """
        Randomly selects

        Need to generate 1min, 5min, 15min, 1hr, 4hr, 1day, 1week"""

        # periods = days * 1440 # minutes
        periods = days * 24  # hours
        # periods = days        # days
        # periods = days / 7    # weeks ?
        price = None

        # randomly pick a distribution function
        dist_func = self.distribution_functions.get(
            random.randint(1, len(self.distribution_functions)))

        # if dist_func != brownian:
        steps = dist_func(loc=0, scale=volatility, size=periods)
        steps[0] = 0
        price = start_price + np.cumsum(steps)
        price = [round(abs(p), 6) for p in price]

        # we will make the code below work but later on
        # else:
        #     # The Wiener process parameter.
        #     delta = 2

        #     total_time = 10.0
        #     num_steps = 500
        #     time_step_size = total_time / num_steps # volatility?
        #     num_realizations = 20 # number of rows

        #     # Create an empty array to store the realizations.
        #     x = np.empty((num_realizations, num_steps + 1))

        #     # Initial values of x.
        #     x[:, 0] = 10_000

        #     bm = brownian(x[:, 0], num_steps, time_step_size, delta, out=x[:, 1:])
        #     price = bm
        #     print(price)

        return pd.DataFrame(
            {
                "ticker": np.repeat([col_name], periods),
                # <------ TEST THIS WITH 'D' INSTEAD OF 'H' <------
                # I think we are doing this because the original example used 'H' (hours) as the smallest unit of measurement
                "date": np.tile(
                    pd.date_range(self.start_date, periods=periods, freq="H"), 1),
                "price": (price),
            }
        )

    def __downsample_ohlc_data(self, timeframe: str) -> None:
        """Converts a higher resolution dataframe into a lower one.
        For example: converts 1min candle sticks into 5min candle sticks.

        # A helpful hint
            https://towardsdatascience.com/pandas-resample-tricks-you-should-know-for-manipulating-time-series-data-7e9643a7e7f3

        # some previous answers
            https://stackoverflow.com/questions/36222928/pandas-ohlc-aggregation-on-ohlc-data

        TODO:
            Resample time frame to 1min, 5min, 15min, 1hr, 4hr, 1d, 1w
        """

        self.__df["date"] = pd.to_datetime(self.__df["date"])
        self.__df = self.__df.set_index("date")

        self.__df = self.__df.resample("W").aggregate(
            {
                "open": lambda s: s[0],
                "high": lambda df: self.__df.max(),
                "low": lambda df: self.__df.min(),
                "close": lambda df: self.__df[-1],
            }
        )

    def __create_whale_candles(self) -> None:
        """Returns a modified self.df containing whale values.
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

        self.__df.reset_index(inplace=True)

        for i in range(len(self.__df)):
            if random.randint(1, 100) <= random_chance:
                # assigns a random floating point multiplier between the range of [WHALE_LOWER_MULT, WHALE_UPPER_MULT].
                # By doing this, every time a whale candle is created, the probability
                # of it being stretched with the same ratio as the
                # previous whale candle or even the next whale candle is essentially 0%
                whale_mult = random.uniform(WHALE_LOWER_MULT, WHALE_UPPER_MULT)

                self.__df.at[i, "open"] = self.__df.iloc[i]["open"] * whale_mult
                self.__df.at[i, "high"] = self.__df.iloc[i]["high"] * whale_mult
                self.__df.at[i, "low"] = self.__df.iloc[i]["low"] / whale_mult
                self.__df.at[i, "close"] = self.__df.iloc[i]["close"] / whale_mult
        self.__df.set_index("date", inplace=True)

    def __extend_wicks(self, hl_mult: float) -> None:
        """Returns a dataframe with the high and low wicks multiplied by the passed in hl_mult"""
        self.__df.reset_index(inplace=True)
        for i in range(len(self.__df)):
            new_h = self.__df.iloc[i]["high"] * hl_mult
            new_l = self.__df.iloc[i]["low"] - (self.__df.iloc[i]["low"] * (hl_mult - 1))

            self.__df.at[i, "high"] = new_h
            self.__df.at[i, "low"] = new_l
        self.__df.set_index("date", inplace=True)
        return self.__df

    def __extend_all_wicks_randomly(self) -> None:
        """Returns a dataframe with the highs and lows multiplied by a random float"""
        self.__df.reset_index(inplace=True)
        for i in range(len(self.__df)):
            h_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)
            l_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)

            new_h = self.__df.iloc[i]["high"] * h_mult
            new_l = self.__df.iloc[i]["low"] - (self.__df.iloc[i]["low"] * (l_mult - 1))

            self.__df.at[i, "high"] = new_h
            self.__df.at[i, "low"] = new_l
        self.__df.set_index("date", inplace=True)
        self.__extend_wicks_randomly()

    def __extend_wicks_randomly(self) -> None:
        """Returns a dataframe with the highs, lows multiplied by a random float

        3 possibilities:
            extend only the high
            extend only the low
            extend both
        """

        self.__df.reset_index(inplace=True)
        for i in range(len(self.__df)):
            h_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)
            l_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)

            random_choice = random.randint(1, 3)

            if random_choice == 1:
                # extend only the high
                self.__df.at[i, "high"] = self.__df.iloc[i]["high"] * h_mult
            elif random_choice == 2:
                # extend only the low
                self.__df.at[i, "low"] = self.__df.iloc[i]["low"] - (
                    self.__df.iloc[i]["low"] * (l_mult - 1)
                )
            else:
                # extend both
                self.__df.at[i, "high"] = self.__df.iloc[i]["high"] * h_mult
                self.__df.at[i, "low"] = self.__df.iloc[i]["low"] - (
                    self.__df.iloc[i]["low"] * (l_mult - 1)
                )
        self.__df.set_index("date", inplace=True)
        return self.__df

    def __connect_open_close_candles(self) -> None:
        """Returns a dataframe where every candles close is the next candles open.
        This is needed because cryptocurrencies run 24/7.
        There are no breaks or pauses so each candle is connected to the next candle.
        """
        self.__df.reset_index(inplace=True)

        for i in range(1, len(self.__df)):
            # connects each open and close together
            self.__df.at[i, "open"] = self.__df.iloc[i - 1]["close"]

            min_value = min(
                self.__df.iloc[i]["open"],
                self.__df.iloc[i]["high"],
                self.__df.iloc[i]["close"],
            )

            max_value = max(
                self.__df.iloc[i]["open"],
                self.__df.iloc[i]["low"],
                self.__df.iloc[i]["close"],
            )

            # something went wrong and the low is not the lowest value
            if self.__df.iloc[i]["low"] > min_value:
                # get the difference between the low and the lowest value and subtract it from the low
                self.__df.at[i, "low"] = self.__df.iloc[i]["low"] - abs(
                    min_value - self.__df.iloc[i]["low"]
                )

            # get the difference between the highest value and the high and add it to the high
            if self.__df.iloc[i]["high"] < max_value:
                self.__df.at[i, "high"] = self.__df.iloc[i]["high"] + abs(
                    max_value - self.__df.iloc[i]["high"]
                )

        self.__df.set_index("date", inplace=True)
        return self.__df

    def create_realistic_candles(self) -> None:
        # make the candles look a little bit more real
        if self.__df is None:
            print(
                "You must call create_random_df before calling create_realistic_candles!"
            )
            return

        self.__create_whale_candles()
        self.__extend_all_wicks_randomly()
        self.__connect_open_close_candles()

    def create_random_df(self) -> None:
        """Create a dataframe for random data"""

        self.__df = self.__generate_random_df(
            self.num_days_range,
            self.start_price,
            self.name,
            self.volatility,
        )

        self.__df.index = pd.to_datetime(self.__df.date)

        # 25% to use a brownian motion distribution instead
        # if random.randint(1, 4) == 4:
        #     self.__df["price"] = self.__brownian_motion_distribution()

        # return self.df.price.resample("D").ohlc()
        self.__df = self.__df.price.resample("D").ohlc(_method="ohlc")
