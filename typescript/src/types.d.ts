/**
 * Shared type definitions for the entire app.
 */

export interface Score {
  right: number;
  wrong: number;
}

export interface SessionResults {
  timestamp: string;
  difficulty: string;
  score: Score;
  guesses: [number, string, string][]; 
  // e.g. [[attemptNumber, userGuess, actualPrice], ...]
  metrics: {
    accuracy: number;
    total_attempts: number;
  };
  system_info: any; // could be more structured if you prefer
}

export interface GameSessionRow {
  id?: number;
  timestamp: string;
  difficulty: string;
  correct_guesses: number;
  wrong_guesses: number;
  accuracy: number;
  system_info: string;
}

export interface GuessRow {
  id?: number;
  session_id: number;
  attempt_number: number;
  user_guess: number;
  actual_price: number;
}

export interface OhlcEntry {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  ma30?: number;
}

export interface DataFrame {
  [timestamp: number]: { open: number; high: number; low: number; close: number };
}

export interface TimeFrameData {
  [timeframe: string]: OhlcEntry[];
}

export interface RandomOhlcParams {
  days_needed: number;
  start_price: number;
  volatility: number;
  drift: number;
}
