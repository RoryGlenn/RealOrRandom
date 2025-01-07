/**
 * random_ohlc.ts
 *
 * Generates synthetic OHLC price data across multiple timeframes
 * using Geometric Brownian Motion (GBM).
 */

import { RandomOhlcParams, TimeFrameData, OhlcEntry } from "./types";
import { DateTime } from "luxon";  // For date handling in TS
import * as fs from "fs";         // Just to show usage, if needed
import * as path from "path";     // For file operations if needed

/**
 * RandomOHLC class for generating synthetic price data at minute resolution
 * and resampling to multiple timeframes.
 */
export class RandomOHLC {
  private _daysNeeded: number;
  private _startPrice: number;
  private _volatility: number;
  private _drift: number;

  constructor(params: RandomOhlcParams) {
    this._daysNeeded = params.days_needed;
    this._startPrice = params.start_price;
    this._volatility = params.volatility;
    this._drift = params.drift;
  }

  get daysNeeded(): number {
    return this._daysNeeded;
  }

  get startPrice(): number {
    return this._startPrice;
  }

  get volatility(): number {
    return this._volatility;
  }

  get drift(): number {
    return this._drift;
  }

  /**
   * Generate minute-level synthetic prices using GBM.
   */
  private _generateRandomPrices(numBars: number): number[] {
    const minuteVol = this._volatility / Math.sqrt(525600);
    const minuteDrift = this._drift / 525600;

    const prices: number[] = [this._startPrice];
    for (let i = 1; i < numBars; i++) {
      const shock = this._randn_bm();
      const lastPrice = prices[i - 1];
      const res = lastPrice * Math.exp(
        (minuteDrift - 0.5 * minuteVol ** 2) * 1 + minuteVol * shock
      );
      prices.push(res);
    }
    console.info("Generated minute-level prices via GBM.");
    return prices;
  }

  /**
   * Utility for generating normally distributed randoms (Box-Muller transform).
   */
  private _randn_bm(): number {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); // Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  /**
   * Create a dictionary of timeframe -> OHLC arrays (with timestamps in Unix epoch).
   */
  public generateOhlcData(): TimeFrameData {
    const minutesInDay = 1440;
    const numMinutes = this._daysNeeded * minutesInDay;

    // Generate the random prices
    const randPrices = this._generateRandomPrices(numMinutes);

    // Build minute-level dataframe
    const startDate = DateTime.fromISO("2030-01-01T00:00:00");
    const minuteData: OhlcEntry[] = [];

    for (let i = 0; i < numMinutes; i++) {
      // Each iteration is 1 minute
      const currentTime = startDate.plus({ minutes: i });
      minuteData.push({
        time: Math.floor(currentTime.toSeconds()),
        open: 0, // Will fix below
        high: 0,
        low: 0,
        close: randPrices[i],
      });
    }

    // Convert raw minuteData into minute OHLC
    // We'll assume close is the real price, so we set open/high/low from adjacent bars
    // and then shift open so that it matches previous bar's close
    for (let i = 0; i < minuteData.length; i++) {
      // For simplicity, let's just say open=close=high=low; or you can vary a bit
      minuteData[i].open = i === 0 ? this._startPrice : minuteData[i - 1].close;
      // We can pretend the bar had 0.1% range up and down, for demonstration
      const c = minuteData[i].close;
      minuteData[i].high = c * 1.001;
      minuteData[i].low = c * 0.999;
    }

    // Now resample
    const timeFrames = ["1min", "5min", "15min", "1h", "4h", "1D", "1W", "1ME"];
    const result: TimeFrameData = {};
    timeFrames.forEach((tf) => {
      result[tf] = this._resampleToTimeframe(minuteData, tf);
    });

    return result;
  }

  /**
   * Resample the minute data to a given timeframe.
   * Example timeframes: "1min", "5min", "15min", "1h", "4h", "1D", "1W", "1ME"
   */
  private _resampleToTimeframe(
    data: OhlcEntry[],
    timeframe: string
  ): OhlcEntry[] {
    // Convert timeframe to # of minutes
    const minutesPerTf = this._timeframeToMinutes(timeframe);
    if (!minutesPerTf) return data; // fallback

    const resampled: OhlcEntry[] = [];
    for (let i = 0; i < data.length; i += minutesPerTf) {
      // chunk for each timeframe
      const chunk = data.slice(i, i + minutesPerTf);
      if (!chunk.length) continue;
      const open = chunk[0].open;
      const close = chunk[chunk.length - 1].close;
      const high = Math.max(...chunk.map((c) => c.high));
      const low = Math.min(...chunk.map((c) => c.low));
      const time = chunk[0].time; // earliest timestamp in chunk

      resampled.push({ time, open, high, low, close });
    }
    return resampled;
  }

  /**
   * Convert a timeframe string to number of minutes. 
   * E.g. "15min" -> 15, "1h" -> 60, "1D" -> 1440, etc.
   */
  private _timeframeToMinutes(tf: string): number {
    switch (tf) {
      case "1min": return 1;
      case "5min": return 5;
      case "15min": return 15;
      case "1h": return 60;
      case "4h": return 240;
      case "1D": return 1440;
      case "1W": return 10080;
      case "1ME":
        // approximate 30 days
        return 43200;
      default:
        return 1;
    }
  }
}
