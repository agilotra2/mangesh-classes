import pandas as pd
import numpy as np
import argparse

# Load and prepare data
def load_data(file, year=None):
    df = pd.read_csv(file)
    df['datetime'] = pd.to_datetime(df['Date'])
    df = df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if year:
        df = df[df['datetime'].dt.year == year].reset_index(drop=True)
    return df

# Backtest function
def backtest_strategy(df, debug=False):
    trades = []
    patterns = []
    days_processed = df['datetime'].dt.date.nunique()
    days_pattern = set()
    days_entry = set()

    trailing_buffer = 5  # fixed points after booking

    i = 0
    while i < len(df) - 6:
        c0, c1, c2, c3 = df.iloc[i], df.iloc[i+1], df.iloc[i+2], df.iloc[i+3]

        # Skip late patterns
        if c3.datetime.hour > 15 or (c3.datetime.hour == 15 and c3.datetime.minute > 0):
            i += 1
            continue

        # Pattern detection
        cond_3ll = c1.Low < c0.Low and c2.Low < c1.Low
        cond_inside = c3.High < c2.High and c3.Low > c2.Low
        cond_green = c3.Close > c3.Open

        if cond_3ll and cond_inside and cond_green:
            days_pattern.add(c3.datetime.date())
            patterns.append({
                "datetime": c3.datetime,
                "baby_high": c3.High,
                "baby_low": c3.Low,
                "baby_open": c3.Open,
                "baby_close": c3.Close,
                "pattern_match": True
            })

            # Entry setup
            dd_long = np.sqrt(c3.High) * 0.2611
            entry_price = c3.High + dd_long
            initial_sl = c3.Low

            entry_candle_index = None
            sl_hit_before_entry = False

            # Entry window (next 3 candles)
            for j in range(i+4, i+7):
                candle = df.iloc[j]
                if candle.Low < initial_sl:
                    sl_hit_before_entry = True
                    break
                if candle.High >= entry_price:
                    entry_candle_index = j
                    break

            if entry_candle_index is not None and not sl_hit_before_entry:
                days_entry.add(c3.datetime.date())
                entry_candle = df.iloc[entry_candle_index]
                entry_time = entry_candle.datetime
                entry_fill_price = entry_price

                # Profit target = max(10 pts, 0.2% entry)
                pct_target_pts = entry_fill_price * 0.0020
                profit_target_points = max(10, pct_target_pts)
                next_target = entry_fill_price + profit_target_points

                tsl = None  # TSL inactive until booking
                booking_made = False
                max_fav_price = entry_fill_price

                if debug:
                    print(f"\nENTRY: {entry_time} @ {entry_fill_price:.2f}, "
                          f"SL={initial_sl:.2f}, Target1={next_target:.2f}")

                j = entry_candle_index + 1
                exit_price, exit_reason = None, None

                while j < len(df):
                    candle = df.iloc[j]

                    # Before booking: normal SL
                    if not booking_made and candle.Low <= initial_sl:
                        exit_price = initial_sl
                        exit_reason = "Stop Loss Hit"
                        break

                    # After booking: TSL active
                    if booking_made and tsl is not None and candle.Low <= tsl:
                        exit_price = tsl
                        exit_reason = "Trailing Stop Hit"
                        break

                    # Profit booking check
                    if candle.High >= next_target:
                        booking_made = True
                        if tsl is None:
                            tsl = next_target - trailing_buffer
                        else:
                            tsl = max(tsl, next_target - trailing_buffer)
                        if debug:
                            print(f"BOOKED at {next_target:.2f}, new TSL={tsl:.2f}")
                        next_target += profit_target_points
                        # After booking, check again if TSL hit within same candle
                        if candle.Low <= tsl:
                            exit_price = tsl
                            exit_reason = "Trailing Stop Hit"
                            break

                    j += 1

                # Session end exit if neither hit
                if exit_price is None:
                    exit_price = df.iloc[j-1].Close
                    exit_reason = "Session End"

                trades.append({
                    "entry_time": entry_time,
                    "entry_price": entry_fill_price,
                    "profit_target_points": profit_target_points,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "profit_points": exit_price - entry_fill_price,
                    "profit_booked": booking_made
                })

                i = j
            else:
                i += 1
        else:
            patterns.append({
                "datetime": c3.datetime,
                "baby_high": c3.High,
                "baby_low": c3.Low,
                "baby_open": c3.Open,
                "baby_close": c3.Close,
                "pattern_match": False
            })
            i += 1

    summary = {
        "days_processed": days_processed,
        "days_pattern_detected": len(days_pattern),
        "days_entry_made": len(days_entry)
    }

    return patterns, trades, summary

# CLI execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bottom Reversal Strategy Backtest with TSL")
    parser.add_argument("--file", required=True, help="Path to historical CSV file")
    parser.add_argument("--year", type=int, help="Year to filter data for")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    df = load_data(args.file, year=args.year)
    if df.empty:
        print("No data found for the given filter/year.")
        exit()

    patterns, trades, summary = backtest_strategy(df, debug=args.debug)

    # Format output datetimes
    patterns_df = pd.DataFrame(patterns)
    if not patterns_df.empty:
        patterns_df['datetime'] = patterns_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['entry_time'] = trades_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')

    print("\n--- SUMMARY ---")
    print(summary)

    print("\n--- TRADE DETAILS ---")
    if not trades_df.empty:
        print(trades_df[['entry_time', 'entry_price', 'profit_target_points',
                         'exit_price', 'exit_reason', 'profit_points', 'profit_booked']])
    else:
        print("No trades executed.")

