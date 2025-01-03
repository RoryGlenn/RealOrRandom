"""
Module for generating random OHLC (Open-High-Low-Close) price data using Geometric Brownian Motion.
Provides functionality to create simulated market data across multiple time intervals.
"""

from logging import getLogger
from datetime import datetime
from typing import Dict
import numpy as np
import pandas as pd

pd.options.display.float_format = "{:.2f}".format

logger = getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RandomOHLC:
    """
    A class to generate random OHLC (Open-High-Low-Close) price data for a specified
    number of bars using Geometric Brownian Motion (GBM).

    Supports resampling the generated data into multiple time intervals, such as
    1min, 5min, 15min, 30min, 1H, 4H, 1D, and others.
    """

    def __init__(
        self,
        days_needed: int,
        start_price: float,
        volatility: float,
        drift: float,
    ) -> None:
        """
        Initialize the RandomOHLC instance with parameters controlling the data generation.

        Parameters
        ----------
        days_needed : int
            The total number of days for which to generate data.
        start_price : float
            The initial price at the start of the simulation.
        volatility : float
            The volatility factor for price simulation.
        drift : float
            The drift factor for price simulation.
        """
        self._days_needed = days_needed
        self._start_price = start_price
        self._volatility = volatility
        self._drift = drift

    @property
    def days_needed(self) -> int:
        return self._days_needed

    @property
    def start_price(self) -> float:
        return self._start_price

    @property
    def volatility(self) -> float:
        return self._volatility

    @property
    def drift(self) -> float:
        return self._drift

    def _generate_random_prices(self, num_bars: int) -> np.ndarray:
        """
        Generate simulated prices using Geometric Brownian Motion (GBM).

        Parameters
        ----------
        num_bars : int
            The number of bars (time intervals) to simulate.

        Returns
        -------
        np.ndarray
            Simulated price values as a NumPy array.
        """
        # Time step size for each bar
        dt = 1 / num_bars
        # Initialize price series with the starting price
        prices = [self._start_price]

        # Generate prices using GBM formula
        for _ in range(num_bars - 1):
            # Calculate random shock using normal distribution
            shock = np.random.normal(0, 1) * np.sqrt(dt)
            # Apply GBM formula for the next price
            res = prices[-1] * np.exp(
                (self._drift - 0.5 * self._volatility**2) * dt
                + self._volatility * shock
            )
            prices.append(res)
        return np.array(prices)

    def _resample_and_convert_to_unix(
        self, df: pd.DataFrame, time_interval: str
    ) -> pd.DataFrame:
        """
        Resample a DataFrame from 1-minute intervals to a specified timeframe
        and convert the index to Unix timestamps.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to resample.
        time_interval : str
            The desired resampling interval (e.g., '1D', '1H').

        Returns
        -------
        pd.DataFrame
            Resampled DataFrame with the index as Unix timestamps.
        """
        candlebar_aggregations = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        # Resample data to the specified interval
        resampled = (
            df.resample(time_interval).aggregate(candlebar_aggregations).round(2)
        )
        # Convert the index from datetime to Unix timestamp
        resampled.index = resampled.index.map(lambda ts: int(ts.timestamp()))
        return resampled

    def _create_timeframe_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create OHLC data for multiple time intervals by resampling.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing minute-level OHLC data.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary mapping timeframe names to resampled DataFrames.
        """
        # Define desired time intervals
        time_intervals = ['15min', '1h', '4h', '1D', '1W', '1ME']
        # Resample the data for each timeframe
        return {
            timeframe: self._resample_and_convert_to_unix(df, timeframe)
            for timeframe in time_intervals
        }

    def generate_ohlc_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate OHLC price data using GBM and resample into various time intervals.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary containing OHLC data for each timeframe, with indices as Unix timestamps.
        """
        # NOTE: We simulate at minute-level resolution to create realistic intraday price movements
        # This allows us to capture authentic high-low variations within each day
        # The minute-level data is then resampled to larger timeframes (15min, 1H, 4H, 1D, 1W, 1ME)
        
        # Calculate required number of minutes to generate the requested number of daily bars
        # For example, if num_bars=30, we need 30 days worth of minute data
        minutes_in_day = 1440
        
        # Each bar represents one day in the final output
        num_minutes = self._days_needed * minutes_in_day

        # Generate random prices using GBM for minute-level resolution
        rand_prices = self._generate_random_prices(num_minutes)

        # Create a datetime range starting from a fixed date
        dt = datetime.strptime("2030-01-01 00:00:00", DATE_FORMAT)
        dates = pd.date_range(start=dt, periods=num_minutes, freq="1min")

        # Create a DataFrame for the simulated prices
        df = pd.DataFrame({"date": dates, "price": rand_prices}).set_index("date")

        # Resample to minute-level OHLC data from the raw prices
        ohlc_data = df["price"].resample("1min").ohlc()

        # Adjust open prices to ensure continuity between candles
        ohlc_data["open"] = ohlc_data["close"].shift(1).fillna(self._start_price)

        # Resample minute-level OHLC to other time intervals
        return self._create_timeframe_data(ohlc_data)
