import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

pd.options.display.float_format = "{:.4f}".format


class RandomOHLC:
    """
    A class to generate random OHLC (Open-High-Low-Close) price data for a specified
    number of days using Geometric Brownian Motion (GBM).

    The class supports resampling the generated data into various timeframes such as
    1min, 5min, 15min, 30min, 1H, 4H, and 1D.
    """

    def __init__(
        self,
        total_days: int,
        start_price: float,
        name: str,
        volatility: float,
        drift: float,
    ) -> None:
        """
        Initialize the RandomOHLC class with parameters controlling the data generation.

        Parameters
        ----------
        total_days : int
            The total number of days for which to generate data.
        start_price : float
            The initial starting price for the simulation.
        name : str
            A name identifier for the generated asset data.
        volatility : float
            The volatility factor applied to the price simulation.
        drift : float
            The drift factor applied to the price simulation.
        """
        self._total_days: int = total_days
        self._start_price: float = start_price
        self._name: str = name
        self._volatility: float = volatility
        self._drift: float = drift

        self._resampled_data: Dict[str, Optional[pd.DataFrame]] = {
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

        self._agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }

    def geometric_brownian_motion(
        self, start_price: float, num_steps: int, drift: float, volatility: float
    ) -> np.ndarray:
        """
        Simulate prices using Geometric Brownian Motion (GBM).

        Parameters
        ----------
        start_price : float
            The initial price of the simulation.
        num_steps : int
            The number of time steps to simulate.
        drift : float
            The drift factor influencing the trend of the price.
        volatility : float
            The volatility factor influencing the randomness of price changes.

        Returns
        -------
        np.ndarray
            An array of simulated prices following a GBM process.
        """
        dt = 1 / num_steps
        prices = [start_price]

        for _ in range(num_steps - 1):
            shock = np.random.normal(0, 1) * np.sqrt(dt)
            prices.append(
                prices[-1]
                * np.exp((drift - 0.5 * (volatility**2)) * dt + volatility * shock)
            )

        return np.array(prices)

    def generate_random_df(
        self, num_bars: int, start_price: float, volatility: float
    ) -> pd.DataFrame:
        """
        Generate daily OHLC price data using GBM.

        This method creates a DataFrame containing simulated daily OHLC data derived from
        per-minute generated prices. The per-minute data is resampled to 1D, providing
        daily open, high, low, and close values.

        Parameters
        ----------
        num_bars : int
            The number of daily bars (days) to generate.
        start_price : float
            The initial price to start the simulation.
        volatility : float
            The volatility factor for the GBM simulation.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by date (daily) with columns: open, high, low, close.
        """
        # Convert days to minutes for simulation
        num_minutes = num_bars * 1440

        prices = self.geometric_brownian_motion(
            start_price=start_price,
            num_steps=num_minutes,
            drift=self._drift,
            volatility=volatility,
        )

        df = pd.DataFrame(
            {
                "date": pd.date_range(start=datetime.now(), periods=num_minutes, freq="1min"),
                "price": prices,
            }
        ).set_index("date")

        # Resample to 1min OHLC from the per-minute prices
        result = df["price"].resample("1min").ohlc()

        # Adjust open prices to the previous candle's close to create continuity
        result["open"] = result["close"].shift(1).fillna(self._start_price)

        # Finally, resample to daily (1D) OHLC data
        result = result.resample("1D").aggregate(self._agg_dict).round(2)
        return result

    def _downsample_ohlc_data(self, timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample OHLC data to a lower timeframe.

        Parameters
        ----------
        timeframe : str
            The target timeframe for resampling (e.g., "5min", "15min", "1H").
        df : pd.DataFrame
            A DataFrame containing OHLC data at a higher frequency (e.g., 1min).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the OHLC data resampled to the specified timeframe.
        """
        return df.resample(timeframe).aggregate(self._agg_dict)

    def _resample_timeframes(self, df: pd.DataFrame) -> None:
        """
        Resample the initial OHLC data into multiple timeframes.

        This method takes the base 1min OHLC data and resamples it to a variety of
        common timeframes, such as 5min, 15min, 30min, 1H, and 1D.

        Parameters
        ----------
        df : pd.DataFrame
            The base OHLC DataFrame at 1min frequency.

        Returns
        -------
        None
        """
        self._resampled_data["1min"] = df
        timeframes = ["5min", "15min", "30min", "1h", "4h", "1D"]
        for timeframe in timeframes:
            self._resampled_data[timeframe] = self._downsample_ohlc_data(
                timeframe, self._resampled_data["1min"]
            )
