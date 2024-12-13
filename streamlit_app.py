import logging
import random
from streamlit.components.v1 import html
import json
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


with open("chart-template.html", "r") as file:
    html_template = file.read()


class GameState:
    """
    Enumeration of the game's possible states.

    Attributes
    ----------
    READY_TO_PLAY : int
        The starting state of the game, before the player begins making guesses.
    WAITING_FOR_GUESS : int
        The state where the game is active and waiting for the user's first guess.
    REVEAL_GUESS_RESULT : int
        The state after the user has submitted a guess, showing whether it was correct
        or wrong, and allowing the user to proceed to the next round.
    GAME_OVER : int
        The state after all attempts have been made, presenting the final results page.
    """

    READY_TO_PLAY: int = -1
    WAITING_FOR_GUESS: int = 0
    REVEAL_GUESS_RESULT: int = 1
    GAME_OVER: int = 2


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


def create_ohlc_df(num_bars: int) -> pd.DataFrame:
    """
    Generate a realistic OHLC dataset for a given number of days.

    Uses a RandomOHLC generator to produce price data, then resamples it to daily (1D)

    Parameters
    ----------
    num_bars : int
        Number of days to generate data for.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing daily OHLC data with columns: "open", "high", "low", "close".
    """

    start_price = 10_000
    volatility = random.uniform(1, 3)
    drift = random.uniform(1, 3)

    rand_ohlc = RandomOHLC(
        num_bars=num_bars,
        start_price=start_price,
        volatility=volatility,
        drift=drift,
    )

    df = rand_ohlc.generate_ohlc_df()

    # for col in df.columns:
    #     df[col] = df[col].clip(lower=1.0)

    logger.info(
        "Num Days: %d, Start Price: %d, Volatility: %.2f, Drift: %.2f",
        num_bars,
        start_price,
        volatility,
        drift,
    )
    return df


def money_to_float(money_str: str) -> float:
    """
    Convert a money-formatted string into a float.

    Parameters
    ----------
    money_str : str
        The money-formatted string (e.g., "$1,234.56").

    Returns
    -------
    float
        The original float value (e.g., 1234.56).
    """
    # Remove "$" and "," then convert to float
    return float(money_str.replace("$", "").replace(",", ""))


def prepare_new_round() -> None:
    """
    Prepare data and state for a new prediction round.

    Dynamically generates OHLC data based on difficulty, selects a future price, and
    creates a set of possible choices for the user to guess from.
    """
    # Map difficulty to required parameters
    difficulty_settings = {
        "Easy": {"extra_bars": 1, "future_offset": 1},
        "Medium": {"extra_bars": 7, "future_offset": 7},
        "Hard": {"extra_bars": 30, "future_offset": 30},
    }
    difficulty = st.session_state.difficulty
    extra_bars = difficulty_settings[difficulty]["extra_bars"]
    future_offset = difficulty_settings[difficulty]["future_offset"]

    # Calculate the number of bars to generate
    num_bars = 90 + extra_bars  # 90 for display + extra bars for future prediction

    # Generate OHLC data
    df = create_ohlc_df(num_bars=num_bars)
    num_display_bars = 90  # Always display the last 90 bars

    # Determine the future bar index
    future_bar_index = num_display_bars + future_offset - 1

    # Fetch future price
    future_price = df["close"].iloc[future_bar_index]

    # Prepare historical data for display
    st.session_state.data = df.iloc[:num_display_bars]

    # Create choices (formatted as money strings)
    choices = sorted(
        [future_price]
        + [round(future_price * (1 + random.uniform(-0.1, 0.1)), 2) for _ in range(3)]
    )
    choices = [f"${c:,.2f}" for c in choices]
    future_price = f"${future_price:,.2f}"

    # Update session state
    st.session_state.future_price = future_price
    st.session_state.choices = choices
    st.session_state.user_choice = None


def create_candlestick_chart(data) -> None:
    """
    Render a candlestick chart using the Lightweight Charts library.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing OHLC data with columns "open", "high", "low", "close".

    Returns
    -------
    None
    """

    # Convert the DataFrame to a format suitable for Lightweight Charts
    candlestick_data = [
        {
            "time": index.strftime("%Y-%m-%d"),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for index, row in data.iterrows()
    ]

    html_content = html_template.replace(
        "candlestick_data", json.dumps(candlestick_data)
    )

    # Use Streamlit's HTML rendering
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

    # Evaluate result
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
    Callback for the back to start button.

    Moves the game to the INITIAL state and prepares a new round.
    """
    st.session_state.clear()
    initialize_session_state()


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
    chart = create_candlestick_chart(st.session_state.data)
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
