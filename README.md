```markdown
# Stock Price Prediction Game

This Streamlit-based application challenges users to predict future stock closing prices after reviewing the last 90 days of generated OHLC (Open-High-Low-Close) data. By simulating realistic price movements and volatility via Geometric Brownian Motion (GBM), the app provides a fun and educational environment to test your market intuition.

## Features

- **Realistic Data Generation:**  
  Utilizes a custom `RandomOHLC` class to generate minute-level price data and then aggregates it into daily OHLC bars for display.

- **Difficulty Levels:**  
  - **Easy:** Predict the closing price 1 day in the future.
  - **Medium:** Predict the closing price 7 days in the future.
  - **Hard:** Predict the closing price 30 days in the future.

- **Game States:**  
  - **START:** Prompting the user to select difficulty and begin.
  - **INITIAL:** Displaying past 90 days of data, waiting for the user's first guess.
  - **SHOW_RESULT:** After a guess, showing correctness and allowing progression.
  - **FINISHED:** Displaying final results after 5 attempts.

- **Dynamic Scoring and Feedback:**  
  Shows correct/wrong counts, calculates accuracy, uses a progress bar, provides encouraging messages, and compares performance to a hypothetical average. Also plots guess correctness over attempts.

- **Interactive Candlestick Chart:**  
  Displays historical prices, helping users spot patterns before making their predictions.

## How It Works

1. **Data Simulation:**  
   The application uses minute-level simulations (via GBM) to produce realistic intraday volatility. This minute data is then resampled into daily OHLC bars to provide more authentic daily patterns.

2. **User Prediction:**  
   Based on difficulty, the user selects a future closing price from several options. The chosen difficulty determines how far into the future the guess projects.

3. **Scoring & Final Results:**  
   Each guess updates the user's score. After 5 attempts, the results page shows accuracy, average error, and a chart comparing guesses to actual prices.

## Getting Started

### Prerequisites
- **Python 3.8+**
- **pip** for installing dependencies

### Installation
```bash
git clone https://github.com/yourusername/stock-prediction-game.git
cd stock-prediction-game
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run streamlit_app.py
```
Then open `http://localhost:8501` in your browser.

## Gameplay Instructions

1. **Start Page:**  
   Choose a difficulty (Easy, Medium, Hard) and press **Start Game**.

2. **Make a Guess:**  
   Review the candlestick chart of the past 90 days. Select a predicted future closing price and submit.

3. **View Feedback:**  
   After submitting, see if you were correct. Continue until all attempts are completed.

4. **Final Results:**  
   Review your performance metrics (accuracy, average error) and a chart showing your guesses vs. actual prices. Use this feedback to improve in future rounds.

## Customization

- **Adjust Days:**  
  Modify `num_bars` in the code to simulate more or fewer days.
  
- **Volatility and Drift:**  
  Adjust the random ranges for volatility and drift to produce different price dynamics.

- **UI Enhancements:**  
  Since it's a Streamlit app, you can easily modify layout, colors, and text.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```