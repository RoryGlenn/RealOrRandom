from typing import Tuple
from datetime import datetime, timedelta

import numpy as np

from constants.constants import *


class Dates:
    @staticmethod
    def get_data_date_ranges() -> dict:
        """Returns a dictionary will the earliest start date and latest end date we can use for each real data file"""
        return {
            # spot
            BINANCE_BTCUSDT_DAY: {"start_date": "2019-09-08", "end_date": "2022-06-16"},
            BINANCE_AAVEUSDT_DAY: {
                "start_date": "2020-10-16",
                "end_date": "2022-06-16",
            },
            BINANCE_ADAUSDT_DAY: {"start_date": "2018-04-17", "end_date": "2022-07-30"},
            BINANCE_CELRUSDT_DAY: {
                "start_date": "2019-03-25",
                "end_date": "2022-07-30",
            },
            BINANCE_DASHUSDT_DAY: {
                "start_date": "2019-03-28",
                "end_date": "2022-07-30",
            },
            BINANCE_DOGEUSDT_DAY: {
                "start_date": "2020-07-10",
                "end_date": "2022-07-30",
            },
            BINANCE_DOTUSDT_DAY: {"start_date": "2020-08-18", "end_date": "2022-07-30"},
            BINANCE_ETCUSDT_DAY: {"start_date": "2018-06-12", "end_date": "2022-07-30"},
            BINANCE_ETHUSDT_DAY: {"start_date": "2017-08-17", "end_date": "2022-07-30"},
            # spot
            BINANCE_ETHUSDT_FUTURES_DAY: {
                "start_date": "2019-11-27",
                "end_date": "2022-03-15",
            },
            BINANCE_LTCUSDT_FUTURES_DAY: {
                "start_date": "2020-01-09",
                "end_date": "2022-03-15",
            },
            BINANCE_ADAUSDT_FUTURES_DAY: {
                "start_date": "2020-01-31",
                "end_date": "2022-07-30",
            },
            BINANCE_BTCUSDT_FUTURES_DAY: {
                "start_date": "2019-09-08",
                "end_date": "2022-03-15",
            },
            BINANCE_XMRUSDT_FUTURES_DAY: {
                "start_date": "2020-02-03",
                "end_date": "2022-07-30",
            },
        }

    @staticmethod
    def get_date_limits(days: int, data_choice: int) -> Tuple[str, str]:
        """Returns the absolute start and end date for a specific data file"""
        d_range = Dates.get_data_date_ranges().get(data_choice)
        start_date__limit_l = [int(i) for i in d_range["start_date"].split("-")]

        # adjust the start_date 91 days after the original start date
        start_dt_limit = datetime(
            year=start_date__limit_l[0],
            month=start_date__limit_l[1],
            day=start_date__limit_l[2],
        )
        adjusted_start_dt = start_dt_limit + timedelta(days=days)
        adj_start_date_limit = adjusted_start_dt.strftime("%Y-%m-%d")
        end_date_limit = d_range["end_date"]
        return adj_start_date_limit, end_date_limit

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
