from os import listdir
from os.path import isfile, join
from sys import exit as sys_exit
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from constants.constants import MINUTES_IN_1DAY, HOURS_IN_1DAY, DATA_PATH, DATE_FORMAT


class RealOHLC:
    def __init__(
        self, data_choice: str, num_days: int, data_files: list[str] = []
    ) -> None:
        self.__num_days = num_days
        self.__data_choice = data_choice
        self.__data_files = data_files
        self.__resampled_data = {}
        self.__df = None
        self.__start_date_limit = None
        self.__end_date_limit = None
        self.__start_date_str = None
        self.__end_date_str = None
        self.__agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }

    @property
    def data_choice(self) -> str:
        return self.__data_choice

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    @property
    def resampled_data(self) -> dict:
        return self.__resampled_data

    @property
    def start_date_dt(self) -> datetime:
        return self.__start_date_limit

    @property
    def end_date_dt(self) -> datetime:
        return self.__end_date_limit

    @property
    def start_date_str(self) -> str:
        return self.__start_date_str

    @property
    def end_date_str(self) -> str:
        return self.__end_date_str

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    def get_filenames(self, path: str) -> list[str]:
        """open the folder containing all the .csv files and return a list containing all the files"""

        return [
            f
            for f in listdir(path)
            if isfile(join(path, f)) and join(path, f)[-4:] == ".csv"
        ]

    def set_start_end_datelimits(self) -> None:
        """Returns a dictionary containing all of the file names as the key
        and start/end dates as the value.
        The start date is also adjusted 90 days into the future
        to avoid out of bounds issues when a random start date is picked.
        """

        # protects loop against incorrectly placed files in the data folder
        if self.__data_choice not in self.__data_files:
            print(f"{self.__data_choice} was not found in data repository")
            sys_exit(1)

        df = pd.read_csv(DATA_PATH + "/" + self.__data_choice, skiprows=1)

        #############################################################################
        # ERROR HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        dt = datetime.strptime(df.loc[len(df) - 1, "date"], DATE_FORMAT)
        # print(dt)
        # dt = dt + timedelta(
        #     days=90
        # )  # why are we moving the start date 90 days forward? I thought we were only supposed to do this with the end date
        # print("adjusted dt", dt)
        #############################################################################

        self.__start_date_limit: str = dt.strftime(DATE_FORMAT)
        self.__end_date_limit: str = df.loc[0, "date"]

    def pick_start_end_dates(self) -> None:
        """Once the start and end date limit have been set,
        Randomly select what our start date will be.
        Then add 'num_days' to the start_date, this will be the end_date."""

        start_date_dt = datetime.strptime(self.__start_date_limit, DATE_FORMAT)
        end_date_dt = datetime.strptime(self.__end_date_limit, DATE_FORMAT)

        # get the number of days from the start date to the end date
        diff_dt = end_date_dt - start_date_dt

        # Create a list of all the dates within the given date bounds.
        # Limit the total number of days we can use to diff_dt.days - num_days
        # so the last 'num_days' will be off limits to the start date.
        # By doing this, we protect ourselves from an out of bounds error
        dt_list = [
            start_date_dt + timedelta(days=x) - timedelta(minutes=1)
            for x in range(diff_dt.days - self.__num_days)
        ]

        if len(dt_list) == 0:
            # dt_list is set incorrectly when self._start_date_limit is out of bounds!!!!
            # start_date_dt      2016-12-16 13:34:00
            # end_date_dt        2016-12-31 23:59:00

            # __start_date_limit 2016-12-16 13:34:00
            # __end_date_limit   2016-12-31 23:59:00

            print("dt_list is empty!")
            print("start_date_dt", self.start_date_dt)
            print("end_date_dt", self.end_date_dt)
            print("__start_date_limit", self.__start_date_limit)
            print("__end_date_limit", self.__end_date_limit)
            sys_exit(1)

        # # randomly choose a start date, then go 'num_days' into the future to get the end date
        start_date_dt = np.random.choice(dt_list)
        end_date_dt = (
            start_date_dt + timedelta(days=self.__num_days) - timedelta(minutes=1)
        )

        # create the start and end date strings
        self.__start_date_str = start_date_dt.strftime(DATE_FORMAT)
        self.__end_date_str = end_date_dt.strftime(DATE_FORMAT)

    def merge_csv_files(self, symbol_pair: str) -> pd.DataFrame:
        """Returns a merged dataframe given a symbol_pair.

        For all files in the data directory, look for the csv files containing
        the symbol_pair and combine them all into one dataframe.
        """
        files = [file for file in self.get_filenames(DATA_PATH) if symbol_pair in file]

        # set the first df
        df_master = pd.read_csv(
            DATA_PATH + "/" + files.pop(0),
            usecols=["date", "open", "high", "low", "close"],
            skiprows=1,
        )[::-1]

        # iterate through the remaining files and concat them to df_master
        for f in files:
            if symbol_pair in f:
                df = pd.read_csv(
                    DATA_PATH + "/" + f,
                    usecols=["date", "open", "high", "low", "close"],
                    skiprows=1,
                )[::-1]
                df_master = pd.concat([df_master, df], ignore_index=True)
        return df_master

    def create_df(self, merge_csvs: bool) -> pd.DataFrame:
        """Create a dataframe for real data"""
        df = None

        if merge_csvs:
            symbol_pair = self.__data_choice.split("_")[1]
            df = self.merge_csv_files(symbol_pair)  # BOTTLE NECK HERE!
        else:
            df = pd.read_csv(
                DATA_PATH + "/" + self.__data_choice,
                usecols=["date", "open", "high", "low", "close"],
                skiprows=1,
            )[::-1]

        print(df)

        df = df.drop(df[df["date"] < self.__start_date_str].index)
        df: pd.DataFrame = df.drop(df[df["date"] > self.__end_date_str].index)

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.set_index("date", inplace=True)
        self.__df = df

        if len(self.__df) != 172_800:
            print(f"create_df: {len(self.__df)} != : 172_800")
            sys_exit(1)

    def normalize_ohlc_data(self) -> pd.DataFrame:
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

        random_multiplier = np.random.randint(9, 999)

        self.__df["open"] = np.round(norm_open * random_multiplier, 4)
        self.__df["high"] = np.round(norm_high * random_multiplier, 4)
        self.__df["low"] = np.round(norm_low * random_multiplier, 4)
        self.__df["close"] = np.round(norm_close * random_multiplier, 4)

    def resample_timeframes(self) -> None:
        """Iterates over all the timeframe keys in resampled_data and creates a
        resampled dataframe corresponding to that timeframe"""

        prev_timeframe = "1min"
        self.__resampled_data["1min"] = self.__df.copy()
        bars_table = self.__create_bars_table()

        for timeframe in bars_table:
            self.__resampled_data[timeframe] = self.__downsample_ohlc_data(
                timeframe, self.__resampled_data[prev_timeframe]
            )
            prev_timeframe = timeframe

    def __downsample_ohlc_data(self, timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a higher resolution dataframe into a lower one.

        For example:
            converts 1min candle sticks into 5min candle sticks.

        The closed parameter controls which end of the interval is inclusive
        while the label parameter controls which end of the interval appears on the resulting index.
        right and left refer to end and the start of the interval, respectively.
        """
        return df.resample(timeframe, label="left", closed="left").aggregate(
            self.__agg_dict
        )

    def __create_bars_table(self) -> dict:
        return {
            "1min": self.__num_days * MINUTES_IN_1DAY,
            "5min": self.__num_days * MINUTES_IN_1DAY // 5,
            "15min": self.__num_days * MINUTES_IN_1DAY // 15,
            "30min": self.__num_days * MINUTES_IN_1DAY // 30,
            "1H": self.__num_days * HOURS_IN_1DAY,
            "2H": self.__num_days * HOURS_IN_1DAY // 2,
            "4H": self.__num_days * HOURS_IN_1DAY // 4,
            "1D": self.__num_days,
            "3D": self.__num_days // 3,
            "1W": self.__num_days // 7,
            "1M": self.__num_days // 30,
        }

    def abstract_dates(self) -> None:
        """Remove the real dates and replace them with fake dates"""
        self.__df.reset_index(inplace=True)

        dates_new = pd.DataFrame(
            {
                "date": np.tile(
                    pd.date_range(
                        start="2000-01-01",
                        periods=len(self.__df),
                        freq="1min",
                    ),
                    1,
                ),
            }
        )

        self.__df["date"] = dates_new["date"]
        self.__df.set_index("date", inplace=True)

    def drop_first_day(self) -> None:
        """For an unknown reason, the first days data is does not show properly in the graph.
        It always generates a very small bar no matter which data set is loaded.

        If we drop the data for the data, the problem goes away, the million dollar question is "why"?
        """
        self.__df.reset_index(inplace=True)
        self.__df.drop([0, 1439], axis=0, inplace=True)
        self.__df.set_index("date", inplace=True)

    def fix_nonsequential_data(self) -> None:
        """Check that we have sequential 1 minute dates.
        If we don't generate it by grabbing the last bars ohlc and copy it to the new bar."""
        from tqdm import tqdm

        # Forward fill / interpolation
        # df = self.__df.interpolate(method="time")

        self.__df.reset_index(inplace=True)
        df = self.__df.copy()

        start_dt = df["date"].iloc[0]

        for i, d in enumerate(tqdm(df["date"])):
            curr_dt = datetime.strptime(str(d), DATE_FORMAT)

            if curr_dt != start_dt:
                print(f"{curr_dt} ?= {start_dt}")
                # curr_ohlc = df.loc[i]

                # get the previous date and time and insert it into the new position
                self.__df.at[i, "date"] = ohlc_prev["date"]
                self.__df.at[i, "open"] = ohlc_prev["open"]
                self.__df.at[i, "high"] = ohlc_prev["high"]
                self.__df.at[i, "low"] = ohlc_prev["low"]
                self.__df.at[i, "close"] = ohlc_prev["close"]
            ohlc_prev = self.__df.loc[i]
            start_dt += timedelta(minutes=1)
        return

    def validate_data(self) -> bool:
        """Confirms whether all data files in data repository have sequential dates"""
        from tqdm import tqdm
        from pprint import pprint

        failed_list = []
        file_count = 1
        print()

        for dfile in self.__data_files:
            df = pd.read_csv(DATA_PATH + "/" + dfile, skiprows=1)[::-1]
            print(f"Checking {dfile} {file_count}\\{len(self.__data_files)}")
            start_dt = datetime.strptime(df["date"].iloc[0], DATE_FORMAT)

            for date in tqdm(df["date"]):
                curr_dt = datetime.strptime(str(date), DATE_FORMAT)

                if curr_dt != start_dt:
                    print(f"{dfile}: {curr_dt} ?= {start_dt}")
                    failed_list.append(f"{dfile}: {curr_dt} ?= {start_dt}")
                    break
                start_dt += timedelta(minutes=1)
            file_count += 1

        pprint(failed_list) if failed_list else print("All files passed")


# DOES PIHOLE ONLY WORK FOR ETHERNET AND NOT WIFI?
