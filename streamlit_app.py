"""
Interactive stock price prediction game using Streamlit.

This module implements a web-based game where users try to predict future stock
prices based on historical OHLC data shown across multiple timeframes.

Functions
---------
initialize_session_state
    Initialize all required Streamlit session state variables.
prepare_new_round
    Generate new OHLC data and setup game parameters for a round.
create_candlestick_chart
    Create and display an interactive multi-timeframe chart.
main
    Main entry point and game flow controller.

Notes
-----
The game supports three difficulty levels:
- Easy: Predict next day's price
- Medium: Predict price 7 days ahead
- Hard: Predict price 30 days ahead
"""

# Standard library imports
from datetime import datetime
from functools import wraps
import json
import logging
import os
import platform
import random
import socket
import time
from typing import Any, Callable, Dict, List

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
from streamlit.components.v1 import html

# Local imports
from database import init_db, store_game_results
from random_ohlc import RandomOHLC

# Initialize Streamlit configuration
st.set_page_config(layout="wide", page_title="Stock Prediction Game")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize database
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error("Failed to initialize database: %s", e)

# Load HTML template
with open("chart-template.html", "r", encoding="utf-8") as file:
    html_template = file.read()

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_system_info() -> Dict[str, Any]:
    """
    Get detailed system information for debugging.

    Returns
    -------
    Dict[str, Any]
        Detailed system information including:
        - OS details
        - CPU information
    """

    try:
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(),
                "hostname": socket.gethostname(),
            },
        }

        logger.info("System info collected successfully")
        return system_info

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error collecting system info: %s", e)
        return {"error": str(e)}


def all_values_same(dict_dfs: dict) -> bool:
    """
    Check if OHLC values are consistent across different timeframes.

    Parameters
    ----------
    dict_dfs : dict
        Dictionary of DataFrames containing OHLC data for different timeframes.

    Returns
    -------
    bool
        True if all values match across timeframes, False otherwise.
    """
    try:
        df_concat = pd.concat(dict_dfs, axis=1, join="inner", keys=dict_dfs.keys())
    except ValueError as e:
        logger.error("Error concatenating DataFrames: %s", e)
        return False

    col_level_names = df_concat.columns.levels[1]  # OHLC column names
    mismatches_by_interval = {}

    for col_name in col_level_names:
        sub_df = df_concat.xs(col_name, axis=1, level=1)
        rowwise_uniques = sub_df.nunique(axis=1)
        mismatches = rowwise_uniques[rowwise_uniques > 1]

        if not mismatches.empty:
            _log_mismatches(mismatches, sub_df, col_name, mismatches_by_interval)
            return False

    logger.info("All OHLC values are consistent across timeframes")
    return True


def _log_mismatches(
    mismatches: pd.Series,
    sub_df: pd.DataFrame,
    col_name: str,
    mismatches_by_interval: Dict[str, Dict[str, Dict[float, List[str]]]],
) -> None:
    """
    Log details about mismatched values across different timeframes.

    Parameters
    ----------
    mismatches : pd.Series
        Series containing indices where mismatches were found.
    sub_df : pd.DataFrame
        DataFrame containing the data for a specific OHLC column.
    col_name : str
        Name of the OHLC column being checked (open, high, low, close).
    mismatches_by_interval : Dict[str, Dict[str, Dict[float, List[str]]]]
        Dictionary to store mismatch information organized by timestamp and column.
    """
    for row_idx in mismatches.index:
        values = sub_df.loc[row_idx].to_dict()
        by_value = {}
        for interval, value in values.items():
            by_value.setdefault(value, []).append(interval)

        if row_idx not in mismatches_by_interval:
            mismatches_by_interval[row_idx] = {}
        mismatches_by_interval[row_idx][col_name] = by_value

    logger.error("Found mismatches in the following timeframes:")
    for timestamp, cols in mismatches_by_interval.items():
        logger.error("At timestamp %s:", timestamp)
        for col, grouped_values in cols.items():
            logger.error("  Column '%s':", col)
            for value, intervals in grouped_values.items():
                logger.error("    Value %.2f in intervals: %s", value, intervals)


class GameState:
    """
    Game state enumeration.

    Represents the different states of game progression.

    Attributes
    ----------
    READY_TO_PLAY : int
        Initial state (-1), showing welcome screen
    WAITING_FOR_GUESS : int
        Active gameplay state (0), waiting for user input
    REVEAL_GUESS_RESULT : int
        Showing result of guess (1)
    GAME_OVER : int
        Final state (2), showing results page
    """

    READY_TO_PLAY: int = -1
    WAITING_FOR_GUESS: int = 0
    REVEAL_GUESS_RESULT: int = 1
    GAME_OVER: int = 2

    @classmethod
    def is_valid_state(cls, state: int) -> bool:
        """Check if a given state is valid."""
        return state in {
            cls.READY_TO_PLAY,
            cls.WAITING_FOR_GUESS,
            cls.REVEAL_GUESS_RESULT,
            cls.GAME_OVER,
        }

    @classmethod
    def get_state_name(cls, state: int) -> str:
        """Get the name of a game state."""
        state_names = {
            cls.READY_TO_PLAY: "Ready to Play",
            cls.WAITING_FOR_GUESS: "Waiting for Guess",
            cls.REVEAL_GUESS_RESULT: "Revealing Result",
            cls.GAME_OVER: "Game Over",
        }
        return state_names.get(state, "Unknown State")


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.

    Parameters
    ----------
    func : Callable
        Function to be timed.

    Returns
    -------
    Callable
        Wrapped function that logs execution time.

    Notes
    -----
    Logs function name and execution time in seconds.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        logger.info(
            "Function '%s' executed in %.4f seconds", func.__name__, execution_time
        )
        return result

    return wrapper


def initialize_session_state() -> None:
    """
    Initialize all required Streamlit session state variables.

    Ensures all necessary game state variables exist and have default values
    before the game starts or continues.

    Notes
    -----
    Initializes the following session state variables:
    - difficulty : str
        Game difficulty level ('Easy', 'Medium', 'Hard')
    - score : Dict[str, int]
        Tracks correct and wrong guesses
    - game_state : GameState
        Current state of the game flow
    - data : Dict[str, pd.DataFrame]
        OHLC data for all timeframes
    - future_price : str
        Target price to predict
    - choices : List[str]
        Available price choices for the user
    - user_choice : Optional[str]
        User's selected price prediction
    - active_interval : str
        Currently displayed timeframe
    - system_info : Dict[str, Any]
        System information for logging
    - guesses : List[Tuple[int, str, str]]
        History of user's guesses
    - msg : Optional[str]
        Current game message to display
    """
    # Initialize all basic variables first
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = "Easy"
    if "score" not in st.session_state:
        st.session_state.score = {"right": 0, "wrong": 0}
    if "data" not in st.session_state:
        st.session_state.data = None
    if "future_price" not in st.session_state:
        st.session_state.future_price = None
    if "choices" not in st.session_state:
        st.session_state.choices = None
    if "user_choice" not in st.session_state:
        st.session_state.user_choice = None
    if "guesses" not in st.session_state:
        st.session_state.guesses = []
    if "msg" not in st.session_state:
        st.session_state.msg = None
    if "active_interval" not in st.session_state:
        st.session_state.active_interval = "1D"
    if "system_info" not in st.session_state:
        st.session_state.system_info = get_system_info()

    # Initialize game state to READY_TO_PLAY
    if "game_state" not in st.session_state:
        st.session_state.game_state = GameState.READY_TO_PLAY


def money_to_float(money_str: str) -> float:
    """
    Convert a money-formatted string into a float.

    Parameters
    ----------
    money_str : str
        String representing a monetary value (e.g., "$1,234.56").

    Returns
    -------
    float
        Numeric value without currency formatting.

    Examples
    --------
    >>> money_to_float("$1,234.56")
    1234.56
    """
    return float(money_str.replace("$", "").replace(",", ""))


def prepare_new_round(start_price: int = 10_000, days_needed: int = 90) -> None:
    """
    Prepare data and state for a new prediction round.

    Generates new OHLC data and sets up game parameters based on the
    current difficulty level.

    Parameters
    ----------
    start_price : int, default 10000
        Starting price for the new data series.
    days_needed : int, default 90
        Number of days of historical data to generate.

    Notes
    -----
    The function:
    1. Adjusts data generation based on difficulty level:
       - Easy: 1 day ahead
       - Medium: 7 days ahead
       - Hard: 30 days ahead
    2. Creates synthetic OHLC data using RandomOHLC
    3. Selects a future price target
    4. Generates multiple choice options within ±10% of target
    5. Updates session state with new data and choices
    """
    difficulty_settings = {
        "Easy": {"extra_bars": 1, "future_offset": 1},
        "Medium": {"extra_bars": 7, "future_offset": 7},
        "Hard": {"extra_bars": 30, "future_offset": 30},
    }
    difficulty = st.session_state.difficulty
    extra_bars = difficulty_settings[difficulty]["extra_bars"]
    future_offset = difficulty_settings[difficulty]["future_offset"]
    days_needed += extra_bars

    logger.info("Starting new round with difficulty: %s", difficulty)
    logger.info(
        "Total days to generate: %d (display: %d, future: %d)",
        days_needed,
        days_needed - extra_bars,
        future_offset,
    )

    rand_ohlc = RandomOHLC(
        days_needed=days_needed,
        start_price=start_price,
        volatility=random.uniform(1, 3),
        drift=random.uniform(1, 3),
    )

    ohlc_data = rand_ohlc.generate_ohlc_data()

    num_display_bars = rand_ohlc.days_needed - extra_bars
    future_bar_index = num_display_bars + future_offset - 1
    future_price = float(ohlc_data["1D"]["close"].iloc[future_bar_index])

    logger.info(
        "Future price target: $%.2f (index: %d)", future_price, future_bar_index
    )

    choices = sorted(
        [future_price]
        + [round(future_price * (1 + random.uniform(-0.1, 0.1)), 2) for _ in range(3)]
    )
    logger.info("Generated choices: %s", [f"${c:,.2f}" for c in choices])

    st.session_state.data = ohlc_data
    st.session_state.future_price = f"${future_price:,.2f}"
    st.session_state.choices = [f"${c:,.2f}" for c in choices]
    st.session_state.user_choice = None


def convert_df_to_candlestick_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert OHLC DataFrame to list of dictionaries for charting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns and DatetimeIndex.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with format suitable for candlestick charts.
        Each dict contains: time, open, high, low, close, ma30.

    Notes
    -----
    - Calculates 30-day moving average
    - Converts time index to numeric format
    - Ensures all numeric columns are float type
    """
    _df = df.copy()

    # Calculate 30-day MA
    _df["ma30"] = _df["close"].rolling(window=30).mean()

    _df["time"] = _df.index
    numeric_cols = ["open", "high", "low", "close", "ma30"]
    _df[numeric_cols] = _df[numeric_cols].astype(float)
    return _df[["time"] + numeric_cols].to_dict("records")


def create_candlestick_chart(data: Dict[str, pd.DataFrame]) -> None:
    """
    Create and display an interactive multi-timeframe candlestick chart.

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dictionary mapping timeframes to OHLC DataFrames.

    Notes
    -----
    Supported timeframes:
    - 1 minute
    - 5 minute
    - 15 minute
    - 1 hour
    - 4 hour
    - Daily
    - Weekly
    - Monthly

    The chart includes:
    - Candlestick display with purple area underneath
    - Timeframe selection buttons
    - Price tooltips
    - Interactive navigation
    - Time display on x-axis (HH:MM format for intraday)
    - Dynamic bar spacing based on data length
    """
    candlestick_data = {
        timeframe: convert_df_to_candlestick_list(df) for timeframe, df in data.items()
    }

    # Log first and last timestamps for each timeframe
    for timeframe, data_list in candlestick_data.items():
        if data_list:
            logger.info(
                "%s timeframe range: %s to %s (%d bars)",
                timeframe,
                pd.Timestamp(data_list[0]["time"], unit="s"),
                pd.Timestamp(data_list[-1]["time"], unit="s"),
                len(data_list),
            )

    candlestick_dict = {
        "one_minute_data": candlestick_data["1min"],
        "five_minute_data": candlestick_data["5min"],
        "fifteen_minute_data": candlestick_data["15min"],
        "one_hour_data": candlestick_data["1h"],
        "four_hour_data": candlestick_data["4h"],
        "day_data": candlestick_data["1D"],
        "week_data": candlestick_data["1W"],
        "month_data": candlestick_data["1ME"],
    }

    html_content = html_template

    for time_interval, df in candlestick_dict.items():
        html_content = html_content.replace(time_interval, json.dumps(df))

    html(html_content, height=800, width=1600)


def display_score() -> None:
    """
    Display the current game score in the Streamlit interface.

    Shows the number of correct and incorrect price predictions
    made by the user in the current session.

    Notes
    -----
    Uses Streamlit's st.write() to display:
    - Number of correct predictions
    - Number of incorrect predictions
    """
    st.subheader("Score")
    st.write(f"Correct: {st.session_state.score['right']}")
    st.write(f"Wrong: {st.session_state.score['wrong']}")


def submit_callback() -> None:
    """
    Process user's price prediction submission.

    Updates the game state and score based on the user's prediction:
    - Compares user's choice with actual future price
    - Updates score (right/wrong counts)
    - Updates game state (reveal result or game over)
    - Stores guess history

    Notes
    -----
    Game ends after 5 attempts, transitioning to GAME_OVER state.
    Otherwise, transitions to REVEAL_GUESS_RESULT state.
    """
    user_choice = st.session_state.user_choice
    future_price = st.session_state.future_price

    if user_choice is None:
        st.warning("Please select a price before submitting.")
        return

    total_attempts = (
        st.session_state.score["right"] + st.session_state.score["wrong"] + 1
    )
    st.session_state.guesses.append((total_attempts, user_choice, future_price))

    if user_choice == future_price:
        st.session_state.score["right"] += 1
        st.session_state.msg = "Correct!"
    else:
        st.session_state.score["wrong"] += 1
        st.session_state.msg = f"Wrong! The correct answer was {future_price}."

    total_attempts = st.session_state.score["right"] + st.session_state.score["wrong"]
    if total_attempts >= 5:
        st.session_state.game_state = GameState.GAME_OVER
    else:
        st.session_state.game_state = GameState.REVEAL_GUESS_RESULT


def next_callback() -> None:
    """
    Prepare the next round of the game.

    Resets the game state to WAITING_FOR_GUESS and generates
    new price data while maintaining current score.

    Notes
    -----
    Updates session state:
    - game_state : Set to WAITING_FOR_GUESS
    - Triggers new data generation via prepare_new_round()
    """
    st.session_state.game_state = GameState.WAITING_FOR_GUESS
    prepare_new_round()


def start_callback() -> None:
    """
    Initialize a new game session.

    Resets game state and generates new price data based on
    selected difficulty level.

    Notes
    -----
    Updates session state:
    - game_state : Set to WAITING_FOR_GUESS
    - Triggers initial data generation via prepare_new_round()
    """
    st.session_state.game_state = GameState.WAITING_FOR_GUESS
    prepare_new_round()


def pregame_callback() -> None:
    """
    Reset the game to initial state.

    Clears all session state variables and reinitializes them
    to start a fresh game.

    Notes
    -----
    - Clears entire session state
    - Calls initialize_session_state() for fresh setup
    - Returns game to READY_TO_PLAY state
    """
    st.session_state.clear()
    initialize_session_state()


def show_results_page() -> None:
    """
    Display the final game results page.

    Shows:
    - Final score and accuracy
    - Table of all guesses and actual prices
    - Interactive plot comparing guesses to actual prices
    - Performance feedback message
    - Option to start a new game

    Notes
    -----
    - Calculates accuracy as (correct guesses / total attempts) * 100
    - Calculates average absolute error between guesses and actual prices
    - Provides different feedback messages based on performance:
      * > 3 correct: Suggests trying harder difficulty
      * 0 correct: Encourages trying again
      * Otherwise: Offers general encouragement
    - Logs game results to database and backup JSON file
    """
    st.markdown("## Final Results")
    st.write("You have completed 5 attempts.")
    display_score()

    guesses_df = pd.DataFrame(
        st.session_state.guesses, columns=["Attempt", "Your Guess", "Actual Price"]
    )

    guesses_df["Your Guess"] = guesses_df["Your Guess"].apply(money_to_float)
    guesses_df["Actual Price"] = guesses_df["Actual Price"].apply(money_to_float)

    guesses_df["Absolute Error"] = (
        guesses_df["Your Guess"] - guesses_df["Actual Price"]
    ).abs()
    accuracy = (st.session_state.score["right"] / 5) * 100
    avg_error = guesses_df["Absolute Error"].mean()

    st.write(f"**Accuracy:** {accuracy:.2f}%")
    st.write(f"**Average Absolute Error:** {avg_error:.2f}")

    st.write("### Your Attempts")
    st.dataframe(guesses_df)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=guesses_df["Attempt"],
            y=guesses_df["Actual Price"],
            mode="lines+markers",
            name="Actual Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=guesses_df["Attempt"],
            y=guesses_df["Your Guess"],
            mode="lines+markers",
            name="Your Guess",
        )
    )

    fig.update_layout(
        title="Your Guesses vs Actual Prices",
        xaxis_title="Attempt Number",
        yaxis_title="Price",
    )

    st.plotly_chart(fig, use_container_width=True)

    message = (
        "**Great job!** You got most of them right. Consider trying a harder "
        "difficulty next time!"
        if st.session_state.score["right"] > 3
        else (
            "**Tough luck this time!** Consider trying again to improve your accuracy."
            if st.session_state.score["right"] == 0
            else "You did okay! With a bit more practice, you might do even better."
        )
    )
    st.write(message)

    st.button("Go Back to Start", on_click=pregame_callback)

    # Log results when game is over
    log_game_results()


def log_game_results() -> None:
    """
    Store game results in database and log file.

    Saves the following information:
    - Timestamp of game completion
    - Difficulty level used
    - Final score
    - History of guesses
    - Performance metrics
    - System information

    Notes
    -----
    - Primary storage is in SQLite database via store_game_results()
    - Backup storage in JSON file at logs/game_history.json
    - Creates logs directory if it doesn't exist
    - Appends to existing JSON file if present
    - Handles storage failures gracefully with logging
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "difficulty": st.session_state.difficulty,
        "score": st.session_state.score,
        "guesses": st.session_state.guesses,
        "metrics": {
            "accuracy": (st.session_state.score["right"] / 5) * 100,
            "total_attempts": 5,
        },
        "system_info": st.session_state.system_info,
    }

    # Store in database
    if store_game_results(results):
        logger.info("Game results stored in database successfully")
    else:
        logger.warning("Failed to store results in database, falling back to JSON")

    # Also keep JSON logging for backup
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/game_history.json"
    try:
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
        history.append(results)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info("Game results logged to JSON file")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to log results to JSON file: %s", e)


def main() -> None:
    """
    Main entry point of the Streamlit app.

    Controls the game flow and user interface based on the current game state:
    - READY_TO_PLAY: Shows welcome screen and difficulty selection
    - GAME_OVER: Shows final results page
    - Other states: Shows game interface with chart and controls

    Notes
    -----
    Game Interface Components:
    - Multi-timeframe candlestick chart
    - Current score display
    - Price prediction interface
    - Submit/Next buttons based on state
    - Success/Error messages for predictions

    Game Flow:
    1. Welcome screen with difficulty selection
    2. Main game loop with price predictions
    3. Results page after 5 attempts
    4. Option to start new game
    """
    initialize_session_state()

    if st.session_state.game_state == GameState.READY_TO_PLAY:
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            st.markdown("## Welcome to the **Ultimate Stock Prediction Challenge**!")
            welcome_text = (
                "You've just joined the analytics team at a top trading firm. "
                "To prove your skills, you'll be shown the last 90 days of a "
                "stock's prices. Your mission? **Predict the future closing price!**"
            )
            st.write(welcome_text)

            st.markdown(
                """
                **Difficulty Levels:**
                - **Easy:** Predict the **next day's** closing price
                - **Medium:** Predict the **closing price 7 days** from now
                - **Hard:** Predict the **closing price 30 days** from now
                """
            )

            game_prompt = (
                "Can you outsmart the market and achieve the highest accuracy "
                "possible? Select a difficulty and press **Start Game** to find out!"
            )
            st.write(game_prompt)

            st.session_state.difficulty = st.radio(
                "Choose your difficulty:", ["Easy", "Medium", "Hard"], index=0
            )
            st.button("Start Game", on_click=start_callback)
        return

    if st.session_state.game_state == GameState.GAME_OVER:
        show_results_page()
        return

    st.title("Stock Price Prediction Game")
    create_candlestick_chart(st.session_state.data)
    display_score()

    st.subheader("What do you think the future closing price will be?")
    st.radio("Choose a price:", st.session_state.choices, key="user_choice")

    if st.session_state.game_state == GameState.WAITING_FOR_GUESS:
        st.button("Submit", on_click=submit_callback)
    elif st.session_state.game_state == GameState.REVEAL_GUESS_RESULT:
        if st.session_state.msg:
            if "Correct" in st.session_state.msg:
                st.success(st.session_state.msg)
            else:
                st.error(st.session_state.msg)

        st.info("Press Next to continue")
        st.button("Next", on_click=next_callback)


if __name__ == "__main__":
    main()
