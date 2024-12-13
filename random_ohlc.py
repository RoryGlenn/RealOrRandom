"""
This module provides a RandomOHLC class to simulate realistic OHLC (Open-High-Low-Close)
stock data over a specified period using Geometric Brownian Motion (GBM). By generating 
per-minute price data and resampling it to daily and other timeframes, the module can 
produce more authentic intraday volatility and daily OHLC patterns.

Classes
-------
RandomOHLC
    Generates random OHLC data for a specified number of bars using GBM. Supports 
    resampling into multiple timeframes (1min, 5min, 15min, 30min, 1H, 2H, 4H, 1D, 
    3D, 1W, 1M) and provides methods for simulating price series, converting minute-level 
    data into daily bars, and further downsampling into various intervals.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:.2f}".format

logger = logging.getLogger(__name__)


class RandomOHLC:
    """
    A class to generate random OHLC (Open-High-Low-Close) price data for a specified
    number of days using Geometric Brownian Motion (GBM).

    The class supports resampling the generated data into various timeframes such as
    1min, 5min, 15min, 30min, 1H, 4H, and 1D.
    """

    def __init__(
        self,
        num_bars: int,
        start_price: float,
        volatility: float,
        drift: float,
    ) -> None:
        """
        Initialize the RandomOHLC class with parameters controlling the data generation.

        Parameters
        ----------
        num_bars : int
            The total number of bars for which to generate data.
        start_price : float
            The initial starting price for the simulation.
        volatility : float
            The volatility factor applied to the price simulation.
        drift : float
            The drift factor applied to the price simulation.
        """
        self._num_bars = num_bars
        self._start_price = start_price
        self._volatility = volatility
        self._drift = drift
        self._agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        self.timeframe_data = {
            "1min": None,
            "5min": None,
            "15min": None,
            "30min": None,
            "1H": None,
            "2H": None,
            "4H": None,
            "1D": None,
            "3D": None,
            "1W": None,
            "1M": None,
        }

    def _generate_random_prices(self, num_bars: int) -> np.ndarray:
        """
        Simulate prices using Geometric Brownian Motion (GBM).

        Parameters
        ----------
        num_bars : int
            The number of candle bars to simulate.

        Returns
        -------
        np.ndarray
            An array of simulated prices following a GBM process.
        """
        dt = 1 / num_bars
        prices = [self._start_price]

        for _ in range(num_bars - 1):
            shock = np.random.normal(0, 1) * np.sqrt(dt)
            res = prices[-1] * np.exp(
                (self._drift - 0.5 * (self._volatility**2)) * dt
                + self._volatility * shock
            )
            prices.append(res)
        return prices

    def generate_ohlc_df(self) -> pd.DataFrame:
        """
        Generate daily OHLC price data using GBM.

        This method creates a DataFrame containing simulated daily OHLC data derived from
        per-minute generated prices. The per-minute data is resampled to 1D, providing
        daily open, high, low, and close values.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by date (daily) with columns: open, high, low, close.
        """

        # NOTE: Why are we converting days to minutes only to resample to 1D?
        # The function first simulates prices at a minute-level resolution to create a realistic intraday price structure, and then resamples that data back into daily bars.
        # This approach is taken because directly generating daily bars from a model like GBM would yield only one price per day, lacking the daily high-low variation.
        # By starting at the minute level, the code can capture intraday volatility and price extremes (opening levels, highest high, lowest low, and closing levels) in a more authentic manner.
        # Once these finer-grained price movements are modeled, the data is resampled to daily intervals, preserving the realistic daily OHLC patterns derived from the high-frequency (minute-level) simulation.

        # Convert days to minutes for simulation
        num_minutes = self._num_bars * 1440

        # Generate random prices using GBM
        rand_prices = self._generate_random_prices(num_bars=num_minutes)

        # Create a DataFrame with per-minute prices
        dates = pd.date_range(start=datetime.now(), periods=num_minutes, freq="1min")
        df = pd.DataFrame({"date": dates, "price": rand_prices}).set_index("date")

        # Resample to 1min OHLC from the per-minute prices
        result = df["price"].resample("1min").ohlc()

        # Adjust open prices to the previous candle's close to create continuity
        result["open"] = result["close"].shift(1).fillna(self._start_price)

        # Finally, resample to daily (1D) OHLC data
        self.timeframe_data = self.create_timeframe_data(result)
        return self.timeframe_data["1D"]

    def create_timeframe_data(self, df: pd.DataFrame) -> None:
        """
        Resample the initial OHLC data into multiple timeframes.

        This method takes the base 1min OHLC data and resamples it to a variety of
        common timeframes, such as "1h", "4h", "1D", "1W", "1M".

        Parameters
        ----------
        df : pd.DataFrame
            The base OHLC DataFrame at 1min frequency.

        Returns
        -------
        Dict[str, pd.DataFrame]: A dictionary of resampled OHLC dataframes for each timeframe.
        """
        aggregations = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        return {
            timeframe: df.resample(timeframe).aggregate(func=aggregations).round(2)
            for timeframe in ["1h", "4h", "1D", "1W", "1M"]
        }
