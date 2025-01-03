import json
import logging
import random
import time
from functools import wraps
from pprint import pprint
from typing import Callable, Dict, List, Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html

from random_ohlc import RandomOHLC

st.set_page_config(layout="wide", page_title="Stock Prediction Game")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


with open("chart-template.html", "r", encoding="utf-8") as file:
    html_template = file.read()


# TODO
# Fix time frames not showing 1h & 4h


class GameState:
    """
    Constants representing different states of the game flow.
    
    READY_TO_PLAY: Initial state before game starts
    WAITING_FOR_GUESS: Waiting for user's price prediction
    REVEAL_GUESS_RESULT: Showing if guess was correct/incorrect
    GAME_OVER: Game completed after 5 attempts
    """

    READY_TO_PLAY: int = -1
    WAITING_FOR_GUESS: int = 0
    REVEAL_GUESS_RESULT: int = 1
    GAME_OVER: int = 2


def timeit(func: Callable) -> Callable:
    """
    A decorator to measure and log the execution time of a function.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The wrapped function with timing measurement.
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
    Initialize all required session state variables if they do not exist.

    Ensures that all necessary keys are present in the Streamlit session_state
    before the game begins or continues.
    """
    if "score" not in st.session_state:
        st.session_state.score = {"right": 0, "wrong": 0}
    if "game_state" not in st.session_state:
        st.session_state.game_state = GameState.READY_TO_PLAY
    if "data" not in st.session_state:
        st.session_state.data = None
    if "future_price" not in st.session_state:
        st.session_state.future_price = None
    if "choices" not in st.session_state:
        st.session_state.choices = None
    if "user_choice" not in st.session_state:
        st.session_state.user_choice = None
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = "Easy"
    if "guesses" not in st.session_state:
        st.session_state.guesses = []
    if "msg" not in st.session_state:
        st.session_state.msg = None


def money_to_float(money_str: str) -> float:
    """
    Convert a money-formatted string into a float.

    Args:
        money_str (str): A string representing a monetary value (e.g. "$1,234.56").

    Returns:
        float: The numeric value (e.g. 1234.56).
    """
    return float(money_str.replace("$", "").replace(",", ""))


def prepare_new_round(start_price: int = 10_000, days_needed: int = 90) -> None:
    """
    Prepare data and state for a new prediction round.

    Dynamically generates OHLC data based on difficulty, selects a future price, and
    creates a set of possible choices for the user to guess from.
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

    rand_ohlc = RandomOHLC(
        days_needed=days_needed,
        start_price=start_price,
        volatility=random.uniform(1, 3),
        drift=random.uniform(1, 3),
    )

    logger.info(
        "Num Days: %d, Start Price: %d, Volatility: %.2f, Drift: %.2f",
        rand_ohlc.days_needed,
        rand_ohlc.start_price,
        rand_ohlc.volatility,
        rand_ohlc.drift,
    )

    ohlc_data = rand_ohlc.generate_ohlc_data()
    num_display_bars = rand_ohlc.days_needed - extra_bars
    future_bar_index = num_display_bars + future_offset - 1
    future_price = float(ohlc_data["1D"]["close"].iloc[future_bar_index])

    choices = sorted(
        [future_price]
        + [round(future_price * (1 + random.uniform(-0.1, 0.1)), 2) for _ in range(3)]
    )
    choices_list = [f"${c:,.2f}" for c in choices]
    future_price_str = f"${future_price:,.2f}"

    st.session_state.data = ohlc_data
    st.session_state.future_price = future_price_str
    st.session_state.choices = choices_list
    st.session_state.user_choice = None


def convert_df_to_candlestick_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame with a DatetimeIndex and OHLC columns into a list of dictionaries.

    Args:
        df (pd.DataFrame): A DataFrame containing 'open', 'high', 'low', 'close' columns and a DatetimeIndex.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries suitable for candlestick charts.
    """
    _df = df.copy()
    _df["time"] = _df.index
    numeric_cols = ["open", "high", "low", "close"]
    _df[numeric_cols] = _df[numeric_cols].astype(float)
    return _df[["time"] + numeric_cols].to_dict("records")


def create_candlestick_chart(data: Dict[str, pd.DataFrame]) -> None:
    """
    Create and display a candlestick chart using the given data dictionary.

    This function extracts the overlapping date range among all timeframes,
    converts them to candlestick data format, and renders the chart using HTML.

    Args:
        data (Dict[str, pd.DataFrame]): A dictionary with timeframe keys and DataFrames as values.
    """

    candlestick_data = {
        timeframe: convert_df_to_candlestick_list(df) for timeframe, df in data.items()
    }

    candlestick_dict = {
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
    Display the current score (number of correct and wrong guesses) to the user.
    """
    st.subheader("Score")
    st.write(f"Correct: {st.session_state.score['right']}")
    st.write(f"Wrong: {st.session_state.score['wrong']}")


def submit_callback() -> None:
    """
    Callback function for the "Submit" button.

    Checks the user's guess against the future price, updates the score, and changes
    the game state based on the total number of attempts made.
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
    Callback function for the "Next" button.

    Sets the game state to WAITING_FOR_GUESS, and prepares a new round of data.
    """
    st.session_state.game_state = GameState.WAITING_FOR_GUESS
    prepare_new_round()


def start_callback() -> None:
    """
    Callback for the start game button.

    Moves the game to the WAITING_FOR_GUESS state and prepares a new round.
    """
    st.session_state.game_state = GameState.WAITING_FOR_GUESS
    prepare_new_round()


def pregame_callback() -> None:
    """
    Callback for the 'Go Back to Start' button on the results page.

    Resets the session state and returns the game to the initial state.
    """
    st.session_state.clear()
    initialize_session_state()


def show_results_page() -> None:
    """
    Display the final results page after the user has completed all attempts.

    Shows the final score, accuracy, average absolute error, and a chart of guesses vs. actual prices.
    Provides a button to restart the game.
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

    if st.session_state.score["right"] > 3:
        st.write(
            "**Great job!** You got most of them right. Consider trying a harder difficulty next time!"
        )
    elif st.session_state.score["right"] == 0:
        st.write(
            "**Tough luck this time!** Consider trying again to improve your accuracy."
        )
    else:
        st.write("You did okay! With a bit more practice, you might do even better.")

    st.button("Go Back to Start", on_click=pregame_callback)


def main() -> None:
    """
    Main entry point of the Streamlit app.

    Handles the game flow:
    - Start page with difficulty selection
    - Initial game state showing historical prices and waiting for a guess
    - After 5 attempts, display results page
    """
    initialize_session_state()

    if st.session_state.game_state == GameState.READY_TO_PLAY:
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            st.markdown("## Welcome to the **Ultimate Stock Prediction Challenge**!")
            st.write(
                "You've just joined the analytics team at a top trading firm. "
                "To prove your skills, you'll be shown the last 90 days of a stock's prices. "
                "Your mission? **Predict the future closing price!**"
            )

            st.markdown(
                """
                **Difficulty Levels:**
                - **Easy:** Predict the **next day's** closing price
                - **Medium:** Predict the **closing price 7 days** from now
                - **Hard:** Predict the **closing price 30 days** from now
                """
            )

            st.write(
                "Can you outsmart the market and achieve the highest accuracy possible? Select a difficulty and press **Start Game** to find out!"
            )

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
