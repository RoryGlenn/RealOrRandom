from pprint import pprint
from typing import Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from constants.constants import *


class Dates:
    @staticmethod
    def get_filenames(path: str) -> list[str]:
        """open the folder containing all the .csv files and return a list containing all the files"""
        from os import listdir
        from os.path import isfile, join

        return [
            f
            for f in listdir(path)
            if isfile(join(path, f)) and join(path, f)[-4:] == ".csv"
        ]

    @staticmethod
    def get_start_end_dates() -> dict[str, dict[str]]:
        """Returns a dictionary containing all of the file names as the key
        and start/end dates as the value.
        The start date is also adjusted 90 days into the future
        to avoid out of bounds issues when a random start date is picked.
        """
        filenames = Dates.get_filenames(DATA_PATH)
        date_ranges = {}

        for file in filenames:
            df = pd.read_csv(DATA_PATH + "/" + file, skiprows=1)
            date = "date" if "date" in df.columns else "Date"
            dt = datetime.strptime(
                df.loc[len(df) - 1, date], "%Y-%m-%d %H:%M:%S"
            ) + timedelta(days=90)
            date_ranges[file] = {
                "start_date": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": df.loc[0, date],
            }
        return date_ranges

    @staticmethod
    def get_start_end_date_strs(date_ranges: dict, num_days: int) -> Tuple[str, str]:
        from datetime import timedelta

        # get the total number of days we can use
        start_date_dt = datetime.strptime(
            date_ranges["start_date"], "%Y-%m-%d %H:%M:%S"
        )
        end_date_dt = datetime.strptime(date_ranges["end_date"], "%Y-%m-%d %H:%M:%S")

        # get the number of days from the start date to the end date
        diff_dt = end_date_dt - start_date_dt

        # Create a list of all the dates within the given date bounds.
        # Limit the total number of days we can use to diff_dt.days-num_days
        # so the last 'num_days' will be off limits to the start date.
        # By doing this, we protect ourselves from an out of bounds error
        dt_list = [
            start_date_dt + timedelta(days=x) for x in range(diff_dt.days - num_days)
        ]

        # randomly choose a start date, then go 'num_days' into the future to get the end date
        start_date_dt = np.random.choice(dt_list)
        end_date_dt = start_date_dt + timedelta(days=num_days)

        # create the start and end date strings
        start_date_str = start_date_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = end_date_dt.strftime("%Y-%m-%d %H:%M:%S")
        return start_date_str, end_date_str

    @staticmethod
    def create_dates(
        num_days_range: int, start_date_limit: str, end_date_limit: str
    ) -> tuple[str, str]:
        """Randomly pick a start and end date within the given starting and ending bounds"""

        start_date_limit_l = [int(i) for i in start_date_limit.split("-")]
        end_date_limit_l = [int(i) for i in end_date_limit.split("-")]

        start_limit_dt = datetime(
            year=start_date_limit_l[0],
            month=start_date_limit_l[1],
            day=start_date_limit_l[2],
        )

        end_limit_dt = datetime(
            year=end_date_limit_l[0], month=end_date_limit_l[1], day=end_date_limit_l[2]
        )

        # get the number of days from the start date to the end date
        date_range_limit = end_limit_dt - start_limit_dt

        # create a list of all the dates within the given date bounds
        dt_list = [
            start_limit_dt + timedelta(days=x) for x in range(date_range_limit.days)
        ]

        # pick a random day to start minus the given range
        start_i = np.random.randint(0, len(dt_list) - num_days_range)
        end_i = start_i + num_days_range

        start_random_dt = dt_list[start_i]
        end_random_dt = dt_list[end_i]
        return start_random_dt.strftime("%Y-%m-%d"), end_random_dt.strftime("%Y-%m-%d")
