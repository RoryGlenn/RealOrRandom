from datetime import date
from statistics import stdev
from time import perf_counter

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from faker import Faker

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

        self.sd = {'open':0, 'high':0, 'low':0, 'close':0}

    @property
    def resampled_data(self) -> dict:
        return self.__resampled_data

    @staticmethod
    def get_time_elapsed(start_time: float) -> float:
        return round(perf_counter() - start_time, 2)

    def __set_sd_ohlc(self) -> list:
        """Sets the standard deviation of the ohlc data"""
        self.sd['open'] = stdev(self.__df['open'])
        self.sd['high'] = stdev(self.__df['high'])
        self.sd['low'] = stdev(self.__df['low'])
        self.sd['close'] = stdev(self.__df['close'])


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
        prices: np.ndarray = self.start_price + np.cumsum(steps)

        # Create (SECONDS_IN_1DAY * num_periods) number of prices
        prices = np.abs(prices.round(decimals=6))

        """
        # Syntax for np.where

        np.where(
            conditional statement -> bool array,
            series/array/function() / scalar if True,
            series/array/function() / scalar if False,
        )


        For example

        np.where(
            df['status_at_time_of_lead'] == 'None',
            df['current_status'],
            df['status_at_time_of_lead']
        )

        df['status'] = np.where(^^^)
        

        # You can also use .values to expose just the np array in the series obj
        np.where(
            df['status_at_time_of_lead'].values == 'None',
            df['current_status'].values,
            df['status_at_time_of_lead'].values
        )

        df['status'] = np.where(^^^)


        # How to vectorize more than 1 conditional statement

        np.select
        


        """


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

        # print(type(self.__df.iloc[0]['close']))
        # print(dir(self.__df.iloc[0]['close']))


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
        prices = np.abs(prices.round(decimals=6))

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
        """Create more volatile periods and replace the normal ones with them"""
        print("Creating volatile periods...")

        random_chance = random.randint(1, 100)

        for i in tqdm(range(len(self.__df))):
            if random.randint(1, VOLATILE_PROB) < random_chance:
                vol_period_num_bars = random.randint(VOLATILE_PERIOD_MIN, VOLATILE_PERIOD_MAX)

                if vol_period_num_bars < len(self.__df) - i:
                    df_new = self.__generate_random_df(
                        num_bars=vol_period_num_bars,
                        frequency="1Min",
                        start_price=self.__df.iloc[i]["close"],
                        volatility=self.volatility * random.uniform(1.01, 1.02),
                    )

                    for j in range(len(df_new)):
                        self.__df.at[i + j, "open"] = df_new.iloc[j]["open"]
                        self.__df.at[i + j, "high"] = df_new.iloc[j]["high"]
                        self.__df.at[i + j, "low"] = df_new.iloc[j]["low"]
                        self.__df.at[i + j, "close"] = df_new.iloc[j]["close"]

    def __create_volatile_periods_vectorized(self) -> None:
        """"""
        print("Creating volatile periods...")

        random_chance = random.randint(1, 100)

        prev_close = self.__df['close'].shift(1).fillna(0).astype(float)
        prev_date = self.__df['date'].shift(1).fillna(pd.Timestamp('1900'))
        choice_list = [self.__generate_random_df, ]

        for j in range(len(df_new)):
            self.__df.at[i + j, "open"] = df_new.iloc[j]["open"]
            self.__df.at[i + j, "high"] = df_new.iloc[j]["high"]
            self.__df.at[i + j, "low"] = df_new.iloc[j]["low"]
            self.__df.at[i + j, "close"] = df_new.iloc[j]["close"]

        conditions = [
            (random.randint(1, 10_000) < random_chance) & (random.randint(500, MINUTES_IN_1DAY) < len(self.__df))  # - i)
            () & ()
        ]

        np.select(condlist=conditions, choicelist=[])




    def __connect_open_close_candles(self) -> None:
        """Returns a dataframe where every candles close is the next candles open.
        This is needed because cryptocurrencies run 24/7.
        There are no breaks or pauses so each candle is connected to the next candle.
        """
        print("Connecting open and closing candles...")

        self.__df : pd.DataFrame
        self.__df.to_numpy()

        # vectorize this!!!
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

    def create_realistic_ohlc(self) -> None:
        """Process for creating slightly more realistic candles"""
        self.__df.reset_index(inplace=True)
        self.__create_volatile_periods()
        self.__set_sd_ohlc()
        self.__connect_open_close_candles() # this is the problem!!!
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
