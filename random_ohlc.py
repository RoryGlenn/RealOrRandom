"""
Generate synthetic OHLC price data using Geometric Brownian Motion.

This module provides functionality to create simulated market data across multiple
timeframes, starting from minute-level data and resampling up to monthly bars.

Classes
-------
RandomOHLC
    Main class for generating synthetic OHLC data.

Notes
-----
The data generation process uses Geometric Brownian Motion (GBM) with configurable
volatility and drift parameters to create realistic price movements.
"""

# Standard library imports
from datetime import datetime
from logging import getLogger
from typing import Dict

# Third-party imports
import numpy as np
import pandas as pd

# Configure pandas display options
pd.options.display.float_format = "{:.2f}".format

# Configure logging
logger = getLogger(__name__)

# Constants
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RandomOHLC:
    """
    Generate synthetic OHLC price data using Geometric Brownian Motion.

    A class that generates realistic price data at minute-level resolution and
    resamples it into multiple timeframes for multi-timeframe analysis.

    Parameters
    ----------
    days_needed : int
        Number of days of data to generate.
    start_price : float
        Initial price for the simulation.
    volatility : float
        Annual volatility parameter for GBM (typically 1-3).
    drift : float
        Annual drift parameter for GBM (typically 1-3).

    Attributes
    ----------
    days_needed : int
        Number of days being simulated.
    start_price : float
        Initial price of the asset.
    volatility : float
        Annual volatility parameter.
    drift : float
        Annual drift parameter.

    Notes
    -----
    The class generates minute-level data first, then resamples it to larger
    timeframes (1min, 5min, 15min, 1h, 4h, 1D, 1W, 1ME) to ensure realistic
    price movements across all timeframes.
    """

    def __init__(
        self,
        days_needed: int,
        start_price: float,
        volatility: float,
        drift: float,
    ) -> None:
        """
        Initialize the RandomOHLC instance.

        Parameters
        ----------
        days_needed : int
            Number of days of data to generate.
        start_price : float
            Initial price for the simulation.
        volatility : float
            Annual volatility parameter for GBM (typically 1-3).
        drift : float
            Annual drift parameter for GBM (typically 1-3).
        """
        self._days_needed = days_needed
        self._start_price = start_price
        self._volatility = volatility
        self._drift = drift

    @property
    def days_needed(self) -> int:
        """
        Number of days of data to generate.

        Returns
        -------
        int
            Total number of days for the simulation.
        """
        return self._days_needed

    @property
    def start_price(self) -> float:
        """
        Initial price for the simulation.

        Returns
        -------
        float
            Starting price value.
        """
        return self._start_price

    @property
    def volatility(self) -> float:
        """
        Annual volatility parameter for GBM.

        Returns
        -------
        float
            Volatility scaling factor (typically 1-3).
        """
        return self._volatility

    @property
    def drift(self) -> float:
        """
        Annual drift parameter for GBM.

        Returns
        -------
        float
            Drift factor controlling trend direction (typically 1-3).
        """
        return self._drift

    def _generate_random_prices(self, num_bars: int) -> np.ndarray:
        """
        Generate simulated prices using Geometric Brownian Motion.

        Parameters
        ----------
        num_bars : int
            Number of price points to generate (in minutes).

        Returns
        -------
        np.ndarray
            Array of simulated price values.

        Notes
        -----
        Uses minute-scaled volatility and drift parameters:
        - Minute volatility = Annual volatility / sqrt(525600)
        - Minute drift = Annual drift / 525600
        """
        # Scale parameters to per-minute
        minute_vol = self._volatility / np.sqrt(525600)  # Annual vol to per-minute vol
        minute_drift = self._drift / 525600  # Annual drift to per-minute drift

        dt = 1  # Since we're already using per-minute parameters
        prices = [self._start_price]

        logger.info("Generating %d minutes of price data with:", num_bars)
        logger.info("  - Start price: $%.2f", self._start_price)
        logger.info(
            "  - Per-minute volatility: %.6f (annual: %.2f)",
            minute_vol,
            self._volatility,
        )
        logger.info(
            "  - Per-minute drift: %.6f (annual: %.2f)", minute_drift, self._drift
        )

        for _ in range(num_bars - 1):
            shock = np.random.normal(0, 1)
            res = prices[-1] * np.exp(
                (minute_drift - 0.5 * minute_vol**2) * dt + minute_vol * shock
            )
            prices.append(res)

        final_price = prices[-1]
        total_return = (final_price - self._start_price) / self._start_price * 100
        logger.info("Price generation complete:")
        logger.info("  - Final price: $%.2f", final_price)
        logger.info("  - Total return: %.2f%%", total_return)

        return np.array(prices)

    def _resample_and_convert_to_unix(
        self, df: pd.DataFrame, time_interval: str
    ) -> pd.DataFrame:
        """
        Resample minute data to a larger timeframe and convert to Unix timestamps.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing minute-level OHLC data.
        time_interval : str
            Target timeframe for resampling (e.g., '15min', '1h', '1D').

        Returns
        -------
        pd.DataFrame
            Resampled OHLC data with Unix timestamp index.

        Notes
        -----
        Aggregation rules:
        - Open: First price in interval
        - High: Highest price in interval
        - Low: Lowest price in interval
        - Close: Last price in interval
        """

        candlebar_aggregations = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }

        resampled = (
            df.resample(rule=time_interval, closed="left", label="left", offset="0min")
            .agg(candlebar_aggregations)
            .dropna()
            .round(2)
        )

        resampled.index = resampled.index.map(lambda ts: int(ts.timestamp()))
        return resampled

    def _create_timeframe_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create OHLC data for multiple timeframes by resampling.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing minute-level OHLC data.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping timeframe names to resampled DataFrames.
            Keys are timeframe identifiers ('1min', '5min', '15min', '1h', etc.).
            Values are DataFrames with OHLC columns and Unix timestamp index.

        Notes
        -----
        Supported timeframes are: 1min, 5min, 15min, 1h, 4h, 1D, 1W, 1ME.
        """
        # Define desired time intervals
        time_intervals = ["1min", "5min", "15min", "1h", "4h", "1D", "1W", "1ME"]

        # Resample the data for each timeframe
        return {
            timeframe: self._resample_and_convert_to_unix(df, timeframe)
            for timeframe in time_intervals
        }

    def generate_ohlc_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate OHLC price data and resample to multiple timeframes.

        Generates minute-level price data using GBM and resamples it into
        various timeframes for multi-timeframe analysis.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing OHLC data for each timeframe.
            Each DataFrame has columns: [open, high, low, close]
            and Unix timestamp index.

        Notes
        -----
        The process involves:
        1. Generating minute-level prices using GBM
        2. Creating OHLC data from raw prices
        3. Resampling to larger timeframes
        4. Converting datetime index to Unix timestamps
        """
        minutes_in_day = 1440
        num_minutes = self._days_needed * minutes_in_day

        logger.info(
            "Generating %d days of data (%d minutes)", self._days_needed, num_minutes
        )

        # Generate random prices using GBM for minute-level resolution
        rand_prices = self._generate_random_prices(num_minutes)

        # Create a datetime range starting from a fixed date
        dt = datetime.strptime("2030-01-01 00:00:00", DATE_FORMAT)
        dates = pd.date_range(start=dt, periods=num_minutes, freq="1min")
        logger.info("Date range: %s to %s", dates[0], dates[-1])

        # Create a DataFrame for the simulated prices
        df = pd.DataFrame({"date": dates, "price": rand_prices}).set_index("date")

        # Resample to minute-level OHLC data from the raw prices
        ohlc_data = df["price"].resample("1min").ohlc()
        logger.info("Created minute OHLC data with shape: %s", ohlc_data.shape)

        # Adjust open prices to ensure continuity between candles
        ohlc_data["open"] = ohlc_data["close"].shift(1).fillna(self._start_price)

        # Log some statistics about the base data
        logger.info("Base OHLC data statistics:")
        logger.info(
            "  Price range: $%.2f to $%.2f",
            ohlc_data["low"].min(),
            ohlc_data["high"].max(),
        )
        logger.info(
            "  Total price movement: %.2f%%",
            (
                (ohlc_data["close"].iloc[-1] - ohlc_data["open"].iloc[0])
                / ohlc_data["open"].iloc[0]
                * 100
            ),
        )

        return self._create_timeframe_data(ohlc_data)
