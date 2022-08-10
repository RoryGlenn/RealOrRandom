import random
import numpy as np
import pandas as pd
from time import perf_counter
from tqdm import tqdm

from brownian import brownian

from constants.constants import *


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

        self.df_pre = None
        self.df_post = None

        self.resampled_data = {
            "1min": None,
            "5min": None,
            "15Min": None,
            "30Min": None,
            "1H": None,
            "2H": None,
            "4H": None,
            "1D": None,
            "3D": None,
            "1W": None,
            "1M": None,
        }

    def get_time_elapsed(self, start_time: float) -> float:
        return round(perf_counter() - start_time, 2)

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

        start_time = perf_counter()
        print("Normalizing OHLC data..")

        _max = np.max(
            [
                np.max(self.df_post.open),
                np.max(self.df_post.high),
                np.max(self.df_post.low),
                np.max(self.df_post.close),
            ]
        )
        _min = np.min(
            [
                np.min(self.df_post.open),
                np.min(self.df_post.high),
                np.min(self.df_post.low),
                np.min(self.df_post.close),
            ]
        )

        norm_open = (self.df_post.open - _min) / (_max - _min)
        norm_high = (self.df_post.high - _min) / (_max - _min)
        norm_low = (self.df_post.low - _min) / (_max - _min)
        norm_close = (self.df_post.close - _min) / (_max - _min)

        random_multiplier = random.randint(9, 999)

        self.df_post["open"] = round(norm_open * random_multiplier, 4)
        self.df_post["high"] = round(norm_high * random_multiplier, 4)
        self.df_post["low"] = round(norm_low * random_multiplier, 4)
        self.df_post["close"] = round(norm_close * random_multiplier, 4)
        print(f"Finished normalize_ohlc_data: {self.get_time_elapsed(start_time)}")

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
        generates 1min, 5min, 15min, 1hr, 4hr, 1day, 1week"""

        periods = days * SECONDS_IN_1DAY
        price = None

        # randomly pick a distribution function
        dist_func = self.distribution_functions.get(
            random.randint(1, len(self.distribution_functions))
        )

        # if dist_func != brownian:
        steps = dist_func(loc=0, scale=volatility, size=periods)
        steps[0] = 0
        price = start_price + np.cumsum(steps)
        price = [round(abs(p), 6) for p in price]

        # start date is arbitrary
        start_date = "2000-01-01"

        # the smallest unit of measurement used is 1 second
        smallest_freq = "1S"

        return pd.DataFrame(
            {
                "ticker": np.repeat([col_name], periods),
                "date": np.tile(
                    pd.date_range(start_date, periods=periods, freq=smallest_freq), 1
                ),
                "price": (price),
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
        start_time = perf_counter()
        print("Creating whale candles...")

        # probability for creating a whale candle will be from 1-20%
        random_chance = random.randint(1, 20)


        for i in tqdm(range(len(self.df_post))):
            if random.randint(1, 100) <= random_chance:
                # assigns a random floating point multiplier between the range of [WHALE_LOWER_MULT, WHALE_UPPER_MULT].
                # By doing this, every time a whale candle is created, the probability
                # of it being stretched with the same ratio as the
                # previous whale candle or even the next whale candle is essentially 0%
                whale_mult = random.uniform(WHALE_LOWER_MULT, WHALE_UPPER_MULT)

                self.df_post.at[i, "open"] = self.df_post.iloc[i]["open"] * whale_mult
                self.df_post.at[i, "high"] = self.df_post.iloc[i]["high"] * whale_mult
                self.df_post.at[i, "low"] = self.df_post.iloc[i]["low"] / whale_mult
                self.df_post.at[i, "close"] = self.df_post.iloc[i]["close"] / whale_mult
        print(f"Finished __create_whale_candles: {self.get_time_elapsed(start_time)}")

    def __extend_wicks(self, hl_mult: float) -> None:
        """Returns a dataframe with the high and low wicks multiplied by the passed in hl_mult"""
        self.df_post.reset_index(inplace=True)
        for i in range(len(self.df_post)):
            new_h = self.df_post.iloc[i]["high"] * hl_mult
            new_l = self.df_post.iloc[i]["low"] - (
                self.df_post.iloc[i]["low"] * (hl_mult - 1)
            )

            self.df_post.at[i, "high"] = new_h
            self.df_post.at[i, "low"] = new_l
        self.df_post.set_index("date", inplace=True)
        return self.df_post

    def __extend_all_wicks_randomly(self) -> None:
        """Returns a dataframe with the highs and lows multiplied by a random float"""
        start_time = perf_counter()
        print("Extending all wicks randomly...")

        for i in tqdm(range(len(self.df_post))):
            h_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)
            l_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)

            new_h = self.df_post.iloc[i]["high"] * h_mult
            new_l = self.df_post.iloc[i]["low"] - (
                self.df_post.iloc[i]["low"] * (l_mult - 1)
            )

            self.df_post.at[i, "high"] = new_h
            self.df_post.at[i, "low"] = new_l
        self.__extend_wicks_randomly()
        print(
            f"Finished __extend_all_wicks_randomly: {self.get_time_elapsed(start_time)}"
        )

    def __extend_wicks_randomly(self) -> None:
        """Returns a dataframe with the highs, lows multiplied by a random float

        3 possibilities:
            extend only the high
            extend only the low
            extend both
        """

        for i in tqdm(range(len(self.df_post))):
            h_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)
            l_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)

            random_choice = random.randint(1, 3)

            if random_choice == 1:
                # extend only the high
                self.df_post.at[i, "high"] = self.df_post.iloc[i]["high"] * h_mult
            elif random_choice == 2:
                # extend only the low
                self.df_post.at[i, "low"] = self.df_post.iloc[i]["low"] - (
                    self.df_post.iloc[i]["low"] * (l_mult - 1)
                )
            else:
                # extend both
                self.df_post.at[i, "high"] = self.df_post.iloc[i]["high"] * h_mult
                self.df_post.at[i, "low"] = self.df_post.iloc[i]["low"] - (
                    self.df_post.iloc[i]["low"] * (l_mult - 1)
                )
        return self.df_post

    def __connect_open_close_candles(self) -> None:
        """Returns a dataframe where every candles close is the next candles open.
        This is needed because cryptocurrencies run 24/7.
        There are no breaks or pauses so each candle is connected to the next candle.
        """
        start_time = perf_counter()
        print("Connecting open and closing candles...")

        for i in tqdm(range(1, len(self.df_post))):
            self.df_post.at[i, "open"] = self.df_post.iloc[i - 1]["close"]

            min_value = min(
                self.df_post.iloc[i]["open"],
                self.df_post.iloc[i]["high"],
                self.df_post.iloc[i]["close"],
            )

            max_value = max(
                self.df_post.iloc[i]["open"],
                self.df_post.iloc[i]["low"],
                self.df_post.iloc[i]["close"],
            )

            # something went wrong and the low is not the lowest value
            if self.df_post.iloc[i]["low"] > min_value:
                # get the difference between the low and the lowest value and subtract it from the low
                self.df_post.at[i, "low"] = self.df_post.iloc[i]["low"] - abs(
                    min_value - self.df_post.iloc[i]["low"]
                )

            # get the difference between the highest value and the high and add it to the high
            if self.df_post.iloc[i]["high"] < max_value:
                self.df_post.at[i, "high"] = self.df_post.iloc[i]["high"] + abs(
                    max_value - self.df_post.iloc[i]["high"]
                )


        print(
            f"Finished __connect_open_close_candles: {self.get_time_elapsed(start_time)}"
        )

    def create_realistic_ohlc(self) -> None:
        """Process for creating slightly more realistic candles"""
        self.df_post.reset_index(inplace=True)
        self.__create_whale_candles()
        self.__extend_all_wicks_randomly()
        self.__connect_open_close_candles()
        self.df_post.set_index("date", inplace=True)

    def create_df(self) -> None:
        """Creates a dataframe for random data"""

        print("Creating random dataframe...")
        start_time = perf_counter()

        self.df_pre = self.__generate_random_df(
            self.num_days_range,
            self.start_price,
            self.name,
            self.volatility,
        )

        self.df_pre.index = pd.to_datetime(self.df_pre.date)

        # assign the first ohlc df
        self.df_post = self.df_pre.price.resample("1min").ohlc(_method="ohlc")

        print(f"Finished create_df: {self.get_time_elapsed(start_time)}")

        # 25% to use a brownian motion distribution instead
        # if random.randint(1, 4) == 4:
        #     self.__df["price"] = self.__brownian_motion_distribution()

    def resample_timeframes(self) -> None:
        """Iterates over all the timeframe keys in resampled_data and creates a
        resampled dataframe corresponding to that timeframe"""

        start_time = perf_counter()

        print(f"Resampling dataframe for {list(self.resampled_data.keys())}")

        # WHERE IS THE SELF.DF_POST?????????????????????
        # WHY ISN'T IT BEING USED HERE?

        # need to resample df using aggregate functions 

        for timeframe in self.resampled_data:
            self.resampled_data[timeframe] = self.df_pre.price.resample(timeframe).ohlc(
                _method="ohlc"
            )

        print(f"Finished resample_timeframes: {self.get_time_elapsed(start_time)}")

    def drop_dates(self) -> None:
        # [print(key, value) for key, value in self.resampled_data.items()]

        print("Dropping date columns...")
        for timeframe, df in self.resampled_data.items():
            print(timeframe, df)
            self.resampled_data[timeframe].reset_index(inplace=True)
            self.resampled_data[timeframe].drop(columns=["date"], inplace=True)
