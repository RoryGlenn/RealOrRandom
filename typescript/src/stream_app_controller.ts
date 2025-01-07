/**
 * stream_app_controller.ts
 *
 * Contains the main "controller" logic for the Stock Price Prediction game.
 * In a web setting, these can be Express routes. In a CLI or TUI setting, 
 * you can directly call these functions.
 */

import { Request, Response } from "express";
import { RandomOHLC } from "./random_ohlc";
import { initDb, storeGameResults } from "./database";
import { SessionResults } from "./types";
import * as os from "os";
import * as fs from "fs";
import * as path from "path";
import { DateTime } from "luxon";

// Mock session state
interface SessionState {
  difficulty: string;
  score: { right: number; wrong: number };
  data: any;
  futurePrice: string | null;
  choices: string[] | null;
  userChoice: string | null;
  guesses: [number, string, string][];
  msg: string | null;
  gameState: number;
  systemInfo: any;
}

// Example enumerations for game states
export enum GameState {
  READY_TO_PLAY = -1,
  WAITING_FOR_GUESS = 0,
  REVEAL_GUESS_RESULT = 1,
  GAME_OVER = 2,
}

let session: SessionState = {
  difficulty: "Easy",
  score: { right: 0, wrong: 0 },
  data: null,
  futurePrice: null,
  choices: null,
  userChoice: null,
  guesses: [],
  msg: null,
  gameState: GameState.READY_TO_PLAY,
  systemInfo: {},
};

export async function initializeApp(): Promise<void> {
  await initDb(); // initialize the SQLite database
  console.log("Database initialized successfully.");

  // We can gather system info as in Python
  session.systemInfo = getSystemInfo();
  // Additional initialization logic if needed
}

function getSystemInfo() {
  return {
    timestamp: new Date().toISOString(),
    system: {
      platform: process.platform,
      machine: os.arch(),
      processor: os.cpus()[0].model,
      cpu_count: os.cpus().length,
      hostname: os.hostname(),
    },
  };
}

export function pregameCallback(): void {
  // Reset session to initial defaults
  session = {
    difficulty: "Easy",
    score: { right: 0, wrong: 0 },
    data: null,
    futurePrice: null,
    choices: null,
    userChoice: null,
    guesses: [],
    msg: null,
    gameState: GameState.READY_TO_PLAY,
    systemInfo: getSystemInfo(),
  };
}

export function startCallback(): void {
  session.gameState = GameState.WAITING_FOR_GUESS;
  prepareNewRound();
}

export function nextCallback(): void {
  session.gameState = GameState.WAITING_FOR_GUESS;
  prepareNewRound();
}

/**
 * This function replicates prepare_new_round() from Python
 */
export function prepareNewRound(startPrice = 10000, daysNeeded = 90): void {
  const difficultySettings: Record<string, { extra_bars: number; future_offset: number }> = {
    Easy: { extra_bars: 1, future_offset: 1 },
    Medium: { extra_bars: 7, future_offset: 7 },
    Hard: { extra_bars: 30, future_offset: 30 },
  };
  const difficulty = session.difficulty;
  const { extra_bars, future_offset } = difficultySettings[difficulty];
  daysNeeded += extra_bars;

  const rand = new RandomOHLC({
    days_needed: daysNeeded,
    start_price: startPrice,
    volatility: Math.random() * (3 - 1) + 1, // random.uniform(1,3)
    drift: Math.random() * (3 - 1) + 1,
  });

  // Generate data
  const ohlcData = rand.generateOhlcData();
  const numDisplayBars = daysNeeded - extra_bars;
  const futureBarIndex = numDisplayBars + future_offset - 1;

  // We'll assume daily data is in ohlcData["1D"]
  const dayData = ohlcData["1D"] || [];
  const futurePrice = dayData[futureBarIndex]?.close || 10000;

  // Generate choices
  const choicesArr: number[] = [futurePrice];
  for (let i = 0; i < 3; i++) {
    const variation = futurePrice * (1 + Math.random() * 0.2 - 0.1);
    choicesArr.push(Math.round(variation * 100) / 100);
  }
  choicesArr.sort((a, b) => a - b);

  session.data = ohlcData;
  session.futurePrice = `$${futurePrice.toLocaleString("en-US", { minimumFractionDigits: 2 })}`;
  session.choices = choicesArr.map((c) => 
    `$${c.toLocaleString("en-US", { minimumFractionDigits: 2 })}`
  );
  session.userChoice = null;
}

/**
 * Convert money string like "$1,234.56" to float
 */
function moneyToFloat(moneyStr: string): number {
  return parseFloat(moneyStr.replace(/\$/g, "").replace(/,/g, ""));
}

/**
 * This replicates submit_callback() from Python
 */
export function submitCallback(): void {
  const userChoice = session.userChoice;
  const futurePrice = session.futurePrice;
  if (!userChoice || !futurePrice) {
    session.msg = "Please select a price before submitting.";
    return;
  }
  const totalAttempts = session.score.right + session.score.wrong + 1;
  session.guesses.push([totalAttempts, userChoice, futurePrice]);

  if (userChoice === futurePrice) {
    session.score.right += 1;
    session.msg = "Correct!";
  } else {
    session.score.wrong += 1;
    session.msg = `Wrong! The correct answer was ${futurePrice}.`;
  }

  if (totalAttempts >= 5) {
    session.gameState = GameState.GAME_OVER;
  } else {
    session.gameState = GameState.REVEAL_GUESS_RESULT;
  }
}

/**
 * Show final results logic
 */
export async function showResultsPage(): Promise<void> {
  const guessesDf = session.guesses.map(([attempt, yourGuess, actualPrice]) => {
    const yGuessFloat = moneyToFloat(yourGuess);
    const aPriceFloat = moneyToFloat(actualPrice);
    const absError = Math.abs(yGuessFloat - aPriceFloat);
    return { attempt, yourGuess: yGuessFloat, actualPrice: aPriceFloat, absoluteError: absError };
  });

  const accuracy = (session.score.right / 5) * 100;
  const avgError =
    guessesDf.reduce((sum, row) => sum + row.absoluteError, 0) / guessesDf.length;

  // Logging final message
  let message: string;
  if (session.score.right > 3) {
    message =
      "**Great job!** You got most of them right. Consider trying a harder difficulty next time!";
  } else if (session.score.right === 0) {
    message = "**Tough luck this time!** Consider trying again to improve your accuracy.";
  } else {
    message = "You did okay! With a bit more practice, you might do even better.";
  }
  console.log(message);

  // Now log results
  await logGameResults(accuracy);
}

async function logGameResults(accuracy: number): Promise<void> {
  const results: SessionResults = {
    timestamp: new Date().toISOString(),
    difficulty: session.difficulty,
    score: session.score,
    guesses: session.guesses,
    metrics: {
      accuracy,
      total_attempts: 5,
    },
    system_info: session.systemInfo,
  };

  const success = await storeGameResults(results);
  if (success) {
    console.info("Game results stored in database successfully.");
  } else {
    console.warn("Failed to store results in database, falling back to JSON.");
  }

  // Also store in JSON as backup
  try {
    const logsDir = path.join(__dirname, "../logs");
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
    const logFile = path.join(logsDir, "game_history.json");
    let history: SessionResults[] = [];
    if (fs.existsSync(logFile)) {
      history = JSON.parse(fs.readFileSync(logFile, "utf-8"));
    }
    history.push(results);
    fs.writeFileSync(logFile, JSON.stringify(history, null, 2), "utf-8");
    console.info("Game results logged to JSON file.");
  } catch (error) {
    console.error("Failed to log results to JSON file:", error);
  }
}
