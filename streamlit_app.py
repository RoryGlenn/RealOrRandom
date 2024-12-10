"""
This module implements a Streamlit-based stock prediction game that challenges the user 
to guess future stock closing prices after reviewing the past 90 days of simulated OHLC data.

The application:
- Generates realistic daily OHLC data using a random simulation (via the RandomOHLC class).
- Offers three difficulty levels (Easy, Medium, Hard), affecting how far into the future
  the user must predict.
- Manages the game state (start, initial, show result, finished) and user interactions 
  through session state variables.
- Displays charts, provides options for guess submissions, and shows final results 
  including score and accuracy.

Classes
-------
GameState
    Enumeration of the game's possible states.
    
Functions
---------
initialize_session_state()
    Initialize required session state variables if they do not exist.
create_ohlc_df(num_bars: int = 150) -> pd.DataFrame
    Generate a realistic OHLC dataset for a given number of days.
prepare_new_round()
    Prepare data and state for a new prediction round.
create_candlestick_chart(data: pd.DataFrame) -> go.Figure
    Create a candlestick chart from the given OHLC data.
display_score()
    Display the current user score.
submit_callback()
    Callback for the Submit button to check the user's guess.
next_callback()
    Callback for the Next button to proceed to the next round.
start_callback()
    Callback for the Start Game button to begin the game.
show_results_page()
    Display the final results page after all attempts are made.
main()
    Main entry point of the Streamlit app.
"""

import logging
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from random_ohlc import RandomOHLC

st.set_page_config(layout="wide", page_title="Stock Prediction Game")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


class GameState:
    """
    Enumeration of the game's possible states.

    Attributes
    ----------
    START : int
        The starting state of the game, before the player begins making guesses.
    INITIAL : int
        The state where the game is active and waiting for the user's first guess.
    SHOW_RESULT : int
        The state after the user has submitted a guess, showing whether it was correct
        or wrong, and allowing the user to proceed to the next round.
    FINISHED : int
        The state after all attempts have been made, presenting the final results page.
    """

    START: int = -1
    INITIAL: int = 0
    SHOW_RESULT: int = 1
    FINISHED: int = 2


def initialize_session_state() -> None:
    """
    Initialize all required session state variables if they do not exist.

    Ensures that all necessary keys are present in the Streamlit session_state
    before the game begins or continues.
    """
    if "score" not in st.session_state:
        st.session_state.score = {"right": 0, "wrong": 0}
    if "game_state" not in st.session_state:
        st.session_state.game_state = GameState.START
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


def create_ohlc_df(num_bars: int = 150) -> pd.DataFrame:
    """
    Generate a realistic OHLC dataset for a given number of days.

    Uses a RandomOHLC generator to produce price data, then resamples it to daily (1D)
    and ensures all prices remain above 1.0.

    Parameters
    ----------
    num_bars : int, optional
        Number of days to generate data for, by default 150.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing daily OHLC data with columns: "open", "high", "low", "close".
    """
    start_price = 10_000
    volatility = random.uniform(1, 3)
    drift = random.uniform(1, 3)

    logger.info(
        "Num Days: %d, Start Price: %d, Volatility: %.2f, Drift: %.2f",
        num_bars,
        start_price,
        volatility,
        drift,
    )

    rand_ohlc = RandomOHLC(
        num_bars=num_bars,
        start_price=start_price,
        name="StockA",
        volatility=volatility,
        drift=drift,
    )

    df = rand_ohlc.generate_ohlc_df()

    for col in df.columns:
        df[col] = df[col].clip(lower=1.0)
    return df


def prepare_new_round() -> None:
    """
    Prepare data and state for a new prediction round.

    Generates new OHLC data, selects a future price based on the chosen difficulty,
    and creates a set of possible choices for the user to guess from.
    """
    df = create_ohlc_df()
    display_days = 90
    last_displayed_day = display_days - 1

    # Determine future day based on difficulty
    if st.session_state.difficulty == "Easy":
        future_day_index = last_displayed_day + 1
    elif st.session_state.difficulty == "Medium":
        future_day_index = last_displayed_day + 7
    else:  # Hard
        future_day_index = last_displayed_day + 30

    future_price = df["close"].iloc[future_day_index]
    st.session_state.data = df.iloc[:display_days]
    st.session_state.future_price = future_price

    choices = sorted(
        [future_price]
        + [round(future_price * (1 + random.uniform(-0.1, 0.1)), 2) for _ in range(3)]
    )

    # choices = [f"${c:,.2f}" for c in choices]

    st.session_state.choices = choices
    st.session_state.user_choice = None


def create_candlestick_chart(data: pd.DataFrame) -> go.Figure:
    """
    Create a candlestick chart from the given OHLC data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing columns "open", "high", "low", "close",
        indexed by date or time.

    Returns
    -------
    go.Figure
        A Plotly Figure object containing the candlestick chart.
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
            )
        ]
    )
    fig.update_layout(
        title="Historical Stock Prices (Last 90 Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,
        height=800,
        width=1400,
    )
    return fig


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

    # Evaluate result
    if user_choice == future_price:
        st.session_state.score["right"] += 1
        # FIXME: make sure this message is shown at the bottom of the page
        # st.success("Correct!")
        st.session_state.msg = "Correct!"
        logger.info("Correct!")
    else:
        st.session_state.score["wrong"] += 1
        # FIXME: make sure this message is shown at the bottom of the page
        # st.error(f"Wrong! The correct answer was {future_price:.2f}.")
        st.session_state.msg = f"Wrong! The correct answer was {future_price:.2f}."
        logger.info("Wrong! The correct answer was %.2f.", future_price)

    total_attempts = st.session_state.score["right"] + st.session_state.score["wrong"]
    if total_attempts >= 5:
        st.session_state.game_state = GameState.FINISHED
    else:
        st.session_state.game_state = GameState.SHOW_RESULT


def next_callback() -> None:
    """
    Callback function for the "Next" button.

    Sets the game state to INITIAL, and prepares a new round of data.
    """
    st.session_state.game_state = GameState.INITIAL
    prepare_new_round()


def start_callback() -> None:
    """
    Callback for the start game button.

    Moves the game to the INITIAL state and prepares a new round.
    """
    st.session_state.game_state = GameState.INITIAL
    prepare_new_round()


def show_results_page() -> None:
    """
    Display the final results page after the user has completed 5 attempts.

    Shows the final score, accuracy, average absolute error, and a chart of guesses vs. actual prices.
    Also provides a button to restart the game.
    """
    st.markdown("## Final Results")
    st.write("You have completed 5 attempts.")
    display_score()

    guesses_df = pd.DataFrame(
        st.session_state.guesses, columns=["Attempt", "Your Guess", "Actual Price"]
    )
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

    if st.button("Go Back to Start"):
        st.session_state.score = {"right": 0, "wrong": 0}
        st.session_state.game_state = GameState.START
        st.session_state.guesses = []


def main() -> None:
    """
    Main entry point of the Streamlit app.

    Handles the game flow:
    - Start page with difficulty selection
    - Initial game state showing historical prices and waiting for a guess
    - After 5 attempts, display results page
    """
    initialize_session_state()

    if st.session_state.game_state == GameState.START:
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

    if st.session_state.game_state == GameState.FINISHED:
        show_results_page()
        return

    if st.session_state.data is None or (
        st.session_state.game_state == GameState.INITIAL
        and st.session_state.user_choice is None
    ):
        prepare_new_round()

    st.title("Stock Price Prediction Game")
    fig = create_candlestick_chart(st.session_state.data)
    st.plotly_chart(fig, use_container_width=False)
    display_score()

    st.subheader("What do you think the future closing price will be?")
    st.radio("Choose a price:", st.session_state.choices, key="user_choice")

    if st.session_state.game_state == GameState.INITIAL:
        st.button("Submit", on_click=submit_callback)
    elif st.session_state.game_state == GameState.SHOW_RESULT:
        if st.session_state.msg:
            if "Correct" in st.session_state.msg:
                st.success(st.session_state.msg)
            else:
                st.error(st.session_state.msg)

        st.info("Press Next to continue")
        st.button("Next", on_click=next_callback)


if __name__ == "__main__":
    main()
