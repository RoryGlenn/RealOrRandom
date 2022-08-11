from pprint import pprint
from time import perf_counter
from datetime import timedelta, date


import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from faker import Faker

# from brownian import brownian
from constants.constants import *

pd.options.display.float_format = "{:.4f}".format


class RandomOHLC:
    def __init__(
        self,
        total_days: int,
        start_price: float,
        name: str,
        volatility: int,
    ) -> None:

        self.total_days = total_days
        self.start_price = start_price
        self.name = name
        self.volatility = volatility
        self.__distribution_functions = {
            1: np.random.normal,
            2: np.random.laplace,
            3: np.random.logistic,
            # 4: brownian,
        }

        self.__df = None
        self.__resampled_data = {
            # Put these back in when testing is over!
            # "1min": None,
            # "5min": None,
            # "15Min": None,
            # "30Min": None,
            # "1H": None,
            # "2H": None,
            "4H": None,
            "1D": None,
            "3D": None,
            "1W": None,
            "1M": None,
        }

        self.__timeframe_table = {
            "1S": SECONDS_IN_1DAY,
            "1min": MINUTES_IN_1DAY,
            "5min": MINUTES_IN_1DAY // 5,
            "15Min": MINUTES_IN_1DAY // 15,
            "30Min": MINUTES_IN_1DAY // 30,
            "1H": HOURS_IN_1DAY,
            "2H": HOURS_IN_1DAY // 2,
            "4H": HOURS_IN_1DAY // 4,
            "1D": DAYS_IN_1DAY,
            "3D": DAYS_IN_1DAY * 3,
            "1W": DAYS_IN_1DAY * 7,
            "1M": DAYS_IN_1DAY * 30,
        }

    @property
    def resampled_data(self) -> dict:
        return self.__resampled_data

    def get_time_elapsed(self, start_time: float) -> float:
        return round(perf_counter() - start_time, 2)

    def generate_random_date(self) -> str:
        return Faker().date_between(
            start_date=date(year=1990, month=1, day=1), end_date="+1y"
        )

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

    def __brownian_motion_distribution(self, rows: int) -> list[float]:
        """Returns a dataframe with a brownian motion distribution"""
        bm_array = self.__brownian_motion(rows)
        bm_array = [i[0] for i in bm_array]
        return self.__normalize_ohlc_list(bm_array)

    def normalize_ohlc_data(self) -> None:
        """
        Normalize OHLC data with random multiplier.
        normalization formula: (data - min) / (max - min)
        """

        print("Normalizing OHLC data..")

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

        print(self.__df)

    def __normalize_ohlc_list(self, data: list) -> list:
        """Normalize OHLC data with random multiplier
        normalization formula: (data - min) / (max - min)
        """
        random_multiplier = random.randint(9, 999)

        _max = np.max(data)
        _min = np.min(data)

        norm_data: np.ndarray = ((data - _min) / (_max - _min)).tolist()
        return [round(i * random_multiplier, 4) for i in norm_data]

    def create_initial_df(self) -> pd.DataFrame:
        """
        Generates a random dataframe.

        1. Randomly selects a distribution function
        2. Randomly generates prices
        3. Creates a dataframe with columns 'date' and 'price'

        """

        print("Creating initial dataframe...")

        # No matter what time frame is given, everything will be converted to seconds!
        # in the future, lets make this more dynamic so that any time period with any frequency can be given!

        period_size = self.total_days * SECONDS_IN_1DAY

        # randomly pick a distribution function
        dist_func = self.__distribution_functions.get(
            random.randint(1, len(self.__distribution_functions))
        )

        steps = dist_func(scale=self.volatility, size=period_size)
        steps[0] = 0
        prices = self.start_price + np.cumsum(steps)

        # Create (SECONDS_IN_1DAY * num_periods) number of prices
        prices = [np.round(p, decimals=6) for p in tqdm(prices)]

        df = pd.DataFrame(
            {
                "date": np.tile(
                    pd.date_range(
                        start=self.generate_random_date(),
                        periods=period_size,
                        freq="1S",
                    ),
                    1,
                ),
                "price": (prices),
            }
        )

        df.index = pd.to_datetime(df.date)
        self.__df = df.price.resample("1min").ohlc(_method="ohlc")
        print(self.__df)

    def __generate_random_df(
        self,
        num_bars: int,
        frequency: str,
        start_price: float,
        volatility: int,
    ) -> pd.DataFrame:

        dist_func = self.__distribution_functions.get(
            random.randint(1, len(self.__distribution_functions))
        )

        steps = dist_func(scale=volatility, size=num_bars)
        steps[0] = 0
        prices = start_price + np.cumsum(steps)
        prices = [np.round(p, decimals=6) for p in prices]

        df = pd.DataFrame(
            {
                "date": np.tile(
                    pd.date_range(
                        start=self.generate_random_date(),
                        periods=num_bars,
                        freq=frequency,
                    ),
                    1,
                ),
                "price": (prices),
            }
        )
        df.index = pd.to_datetime(df.date)
        df = df.price.resample("1min").ohlc(_method="ohlc")
        return df

    def __create_volatile_periods(self) -> None:
        print("Creating volatile periods...")

        for i in range(tqdm(len(self.__df))):
            if random.randint(1, 1_000) == 1:
                num_bars = random.randint(1, MINUTES_IN_1DAY)
                if num_bars < len(self.__df) - i:
                    df_new = self.__generate_random_df(
                        num_bars=num_bars,
                        frequency="1Min",
                        start_price=self.__df.iloc[i]["close"],
                        volatility=self.volatility * 1.01,
                    )

                    for j in range(len(df_new)):
                        self.__df.at[i + j, "open"] = df_new.iloc[j]["open"]
                        self.__df.at[i + j, "high"] = df_new.iloc[j]["high"]
                        self.__df.at[i + j, "low"] = df_new.iloc[j]["low"]
                        self.__df.at[i + j, "close"] = df_new.iloc[j]["close"]

                        if i+j >= len(self.__df):
                            print(i+j, len(self.__df))
                            print("shits busted")
                            exit(1)
        print(self.__df)

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
        print("Creating whale candles...")

        # probability for creating a whale candle will be from 1%-100%
        random_chance = random.randint(1, 100)

        for i in tqdm(range(len(self.__df))):
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

    def __extend_all_wicks_randomly(self) -> None:
        """Returns a dataframe with the highs and lows multiplied by a random float"""
        print("Extending all wicks randomly...")

        for i in tqdm(range(len(self.__df))):
            h_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)
            l_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)

            new_h = self.__df.iloc[i]["high"] * h_mult
            new_l = self.__df.iloc[i]["low"] - (self.__df.iloc[i]["low"] * (l_mult - 1))

            self.__df.at[i, "high"] = new_h
            self.__df.at[i, "low"] = new_l

    def __extend_wicks_randomly(self) -> None:
        """Returns a dataframe with the highs, lows multiplied by a random float

        3 possibilities:
            extend only the high
            extend only the low
            extend both
        """
        print("Extend wicks randomly...")

        for i in tqdm(range(len(self.__df))):
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

    def __connect_open_close_candles(self) -> None:
        """Returns a dataframe where every candles close is the next candles open.
        This is needed because cryptocurrencies run 24/7.
        There are no breaks or pauses so each candle is connected to the next candle.
        """
        print("Connecting open and closing candles...")

        for i in tqdm(range(1, len(self.__df))):
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
        print(self.__df)

    def create_realistic_ohlc(self) -> None:
        """Process for creating slightly more realistic candles"""
        self.__df.reset_index(inplace=True)
        self.__create_volatile_periods()
        # self.__create_whale_candles()
        # self.__extend_all_wicks_randomly()
        # self.__extend_wicks_randomly()
        self.__connect_open_close_candles()
        self.__df.set_index("date", inplace=True)

    def __downsample_ohlc_data(self, timeframe: str, df: pd.DataFrame) -> None:
        """
        Converts a higher resolution dataframe into a lower one.

        For example:
            converts 1min candle sticks into 5min candle sticks.
        """
        return df.resample(timeframe).aggregate(
            {
                "open": lambda s: s[0],
                "high": lambda df: df.max(),
                "low": lambda df: df.min(),
                "close": lambda df: df[-1],
            }
        )

    def resample_timeframes(self) -> None:
        """Iterates over all the timeframe keys in resampled_data and creates a
        resampled dataframe corresponding to that timeframe"""

        for timeframe in tqdm(self.__resampled_data):
            # since we already resampled self.df_post to 1min,
            # there is no need to resample it again.
            if timeframe == "1min":
                # is the 1min df stored in resampled data?
                self.__resampled_data[timeframe] = self.__df
                continue

            # resample the same df each time
            self.__resampled_data[timeframe] = self.__downsample_ohlc_data(
                timeframe, self.__df
            )

    def drop_dates(self) -> None:
        """Drops the date column on all dataframes"""
        print("Dropping date columns...")

        for timeframe, df in self.__resampled_data.items():
            self.__resampled_data[timeframe].reset_index(inplace=True)
            self.__resampled_data[timeframe].drop(columns=["date"], inplace=True)

    def print_resampled_data(self) -> None:
        {print(tf + "\n", df) for tf, df in self.resampled_data.items()}
