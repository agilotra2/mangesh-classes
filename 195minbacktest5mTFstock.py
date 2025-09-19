import argparse
import pandas as pd
import numpy as np
import os
import shutil
from datetime import time, datetime

DEFAULT_START_EQUITY = 100000

def load_data(file_path, year=None):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.rename(columns={'Date': 'datetime',
                       'Open': 'open', 'High': 'high',
                       'Low': 'low', 'Close': 'close',
                       'Volume': 'volume'}, inplace=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    if year:
        df = df[df['datetime'].dt.year == year]
    return df

def calculate_pivots(df):
    df['date'] = df['datetime'].dt.date
    pivots = {}
    day_groups = list(df.groupby('date'))
    for i in range(len(day_groups) - 1):
        day, day_data = day_groups[i]
        second_half = day_data[
            (day_data['datetime'].dt.time >= time(12, 30)) &
            (day_data['datetime'].dt.time <= time(15, 25))
        ]
        if second_half.empty:
            continue
        high = second_half['high'].max()
        low = second_half['low'].min()
        dd_long = np.sqrt(high) * 0.2611
        dd_short = np.sqrt(low) * 0.2611
        pivots[day] = {
            'second_half_high': high,
            'second_half_low': low,
            'dd_long': dd_long,
            'dd_short': dd_short
        }
    return pivots

def simulate_trades(df, pivots, tsl_trigger, trail_size):
    df['date'] = df['datetime'].dt.date
    results = []
    day_list = sorted(df['date'].unique())

    for idx in range(1, len(day_list)):
        prev_day = day_list[idx - 1]
        curr_day = day_list[idx]
        if prev_day not in pivots:
            continue
        day_data = df[df['date'] == curr_day].copy()

        high_level = pivots[prev_day]['second_half_high'] + pivots[prev_day]['dd_long']
        low_level = pivots[prev_day]['second_half_low'] - pivots[prev_day]['dd_short']
        dd_long = pivots[prev_day]['dd_long']
        dd_short = pivots[prev_day]['dd_short']

        trade_taken = False
        entry_type, entry_price, stop_loss = None, None, None
        target_points, entry_time, exit_time = None, None, None
        pnl = 0
        highest_price, lowest_price = None, None
        tsl_active = False
        next_target = None

        for _, row in day_data.iterrows():
            candle_time, close_price = row['datetime'].time(), row['close']

            if candle_time > time(15, 5) and not trade_taken:
                break

            if not trade_taken:
                if close_price > high_level:
                    entry_type = "LONG"
                    entry_price = round(close_price, 2)
                    stop_loss = round(row['low'] - dd_long, 2)
                    target_points = max(10, entry_price * 0.002)
                    next_target = entry_price + target_points
                    highest_price = row['high']
                    trade_taken, entry_time = True, row['datetime']
                    continue
                elif close_price < low_level:
                    entry_type = "SHORT"
                    entry_price = round(close_price, 2)
                    stop_loss = round(row['high'] + dd_short, 2)
                    target_points = max(10, entry_price * 0.002)
                    next_target = entry_price - target_points
                    lowest_price = row['low']
                    trade_taken, entry_time = True, row['datetime']
                    continue

            else:
                if entry_type == "LONG":
                    highest_price = max(highest_price, row['high'])
                    if not tsl_active and highest_price - entry_price >= tsl_trigger:
                        tsl_active = True
                    if tsl_active:
                        new_tsl = highest_price - trail_size
                        if new_tsl > stop_loss:
                            stop_loss = round(new_tsl, 2)
                    while row['high'] >= next_target:
                        next_target += target_points
                    if row['low'] <= stop_loss:
                        pnl, exit_time = stop_loss - entry_price, row['datetime']
                        break

                elif entry_type == "SHORT":
                    lowest_price = min(lowest_price, row['low'])
                    if not tsl_active and entry_price - lowest_price >= tsl_trigger:
                        tsl_active = True
                    if tsl_active:
                        new_tsl = lowest_price + trail_size
                        if new_tsl < stop_loss:
                            stop_loss = round(new_tsl, 2)
                    while row['low'] <= next_target:
                        next_target -= target_points
                    if row['high'] >= stop_loss:
                        pnl, exit_time = entry_price - stop_loss, row['datetime']
                        break

        if trade_taken:
            if exit_time is None:
                last_close = round(day_data.iloc[-1]['close'], 2)
                pnl = last_close - entry_price if entry_type == "LONG" else entry_price - last_close
                exit_time_str = "EOD"
            else:
                exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M")

            results.append({
                "Date": curr_day,
                "Year": curr_day.year,
                "Type": entry_type,
                "Entry Time": entry_time.strftime("%Y-%m-%d %H:%M"),
                "Entry Price": entry_price,
                "Stop Loss": round(stop_loss, 2),
                "Trail Trigger": tsl_trigger,
                "Trail Size": trail_size,
                "Exit Time": exit_time_str,
                "PnL": round(pnl, 2),
            })

    return pd.DataFrame(results)

def debug_trade_day(df, pivots, tsl_trigger, trail_size, debug_day):
    # Print debug info for that day only
    df['date'] = df['datetime'].dt.date
    day_list = sorted(df['date'].unique())
    debug_day_dt = pd.to_datetime(debug_day).date()

    idx = None
    for i in range(1, len(day_list)):
        curr_day = day_list[i]
        if curr_day == debug_day_dt:
            idx = i
            break

    if idx is None:
        print(f"Specified day {debug_day} not found in the data file.")
        return

    prev_day = day_list[idx - 1]
    curr_day = day_list[idx]
    if prev_day not in pivots:
        print(f"No pivots for previous day {prev_day}. Cannot backtest.")
        return

    day_data = df[df['date'] == curr_day].copy()

    print(f"\n=== Debugging for Stock on {curr_day} ===")
    print(f"Previous Day: {prev_day}")
    print(f"Pivots: {pivots[prev_day]}")

    high_level = pivots[prev_day]['second_half_high'] + pivots[prev_day]['dd_long']
    low_level = pivots[prev_day]['second_half_low'] - pivots[prev_day]['dd_short']
    dd_long = pivots[prev_day]['dd_long']
    dd_short = pivots[prev_day]['dd_short']

    print(f"High Level: {high_level:.2f}")
    print(f"Low Level: {low_level:.2f}")
    print(f"DD Long: {dd_long:.2f}")
    print(f"DD Short: {dd_short:.2f}\n")

    trade_taken = False
    entry_type, entry_price, stop_loss = None, None, None
    target_points, entry_time, exit_time = None, None, None
    pnl = 0
    highest_price, lowest_price = None, None
    tsl_active = False
    next_target = None

    for _, row in day_data.iterrows():
        candle_time, close_price = row['datetime'].time(), row['close']
        debug_line = f"{row['datetime']} | Price: {close_price:.2f}"

        if not trade_taken and candle_time > time(15, 5):
            print(debug_line + " | No trade taken before cutoff.")
            break

        if not trade_taken:
            debug_line += f" | Checking entry..."
            if close_price > high_level:
                entry_type = "LONG"
                entry_price = round(close_price, 2)
                stop_loss = round(row['low'] - dd_long, 2)
                target_points = max(10, entry_price * 0.002)
                next_target = entry_price + target_points
                highest_price = row['high']
                trade_taken, entry_time = True, row['datetime']
                debug_line += f" | LONG ENTRY TRIGGERED @ {entry_price}, SL: {stop_loss}, Target Step: {target_points:.2f}"
            elif close_price < low_level:
                entry_type = "SHORT"
                entry_price = round(close_price, 2)
                stop_loss = round(row['high'] + dd_short, 2)
                target_points = max(10, entry_price * 0.002)
                next_target = entry_price - target_points
                lowest_price = row['low']
                trade_taken, entry_time = True, row['datetime']
                debug_line += f" | SHORT ENTRY TRIGGERED @ {entry_price}, SL: {stop_loss}, Target Step: {target_points:.2f}"
            else:
                debug_line += " | No entry."

            print(debug_line)
            continue

        # Trade ongoing
        debug_line += f" | In Trade ({entry_type})"
        if entry_type == "LONG":
            prev_highest = highest_price
            highest_price = max(highest_price, row['high'])
            debug_line += f" | Highest: {highest_price:.2f}"
            if not tsl_active and highest_price - entry_price >= tsl_trigger:
                tsl_active = True
                debug_line += f" | TSL ACTIVE (Profit: {highest_price-entry_price:.2f} >= {tsl_trigger})"
            if tsl_active:
                new_tsl = highest_price - trail_size
                if new_tsl > stop_loss:
                    debug_line += f" | Trailing SL updated: {stop_loss:.2f} -> {new_tsl:.2f}"
                    stop_loss = round(new_tsl, 2)
            debug_line += f" | SL: {stop_loss:.2f} Next Target: {next_target:.2f}"
            targets_hit = 0
            while row['high'] >= next_target:
                next_target += target_points
                targets_hit += 1
            if targets_hit > 0:
                debug_line += f" | Targets hit: {targets_hit}"
            if row['low'] <= stop_loss:
                pnl = stop_loss - entry_price
                exit_time = row['datetime']
                debug_line += f" | SL HIT, EXIT TRADE @ {stop_loss:.2f} (PnL: {pnl:.2f})"
                print(debug_line)
                break

        elif entry_type == "SHORT":
            prev_lowest = lowest_price
            lowest_price = min(lowest_price, row['low'])
            debug_line += f" | Lowest: {lowest_price:.2f}"
            if not tsl_active and entry_price - lowest_price >= tsl_trigger:
                tsl_active = True
                debug_line += f" | TSL ACTIVE (Profit: {entry_price - lowest_price:.2f} >= {tsl_trigger})"
            if tsl_active:
                new_tsl = lowest_price + trail_size
                if new_tsl < stop_loss:
                    debug_line += f" | Trailing SL updated: {stop_loss:.2f} -> {new_tsl:.2f}"
                    stop_loss = round(new_tsl, 2)
            debug_line += f" | SL: {stop_loss:.2f} Next Target: {next_target:.2f}"
            targets_hit = 0
            while row['low'] <= next_target:
                next_target -= target_points
                targets_hit += 1
            if targets_hit > 0:
                debug_line += f" | Targets hit: {targets_hit}"
            if row['high'] >= stop_loss:
                pnl = entry_price - stop_loss
                exit_time = row['datetime']
                debug_line += f" | SL HIT, EXIT TRADE @ {stop_loss:.2f} (PnL: {pnl:.2f})"
                print(debug_line)
                break
        print(debug_line)

    if not trade_taken:
        print("No trade taken on this day.")
    else:
        if exit_time is None:
            last_close = round(day_data.iloc[-1]['close'], 2)
            pnl = last_close - entry_price if entry_type == "LONG" else entry_price - last_close
            print(f"Trade open until EOD ({entry_type}). PnL: {pnl:.2f}")
        else:
            print(f"Trade exited at {exit_time} with PnL: {pnl:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Backtest Pivot Breakout Strategy (with debug/day mode)")
    parser.add_argument("--dir", help="Directory containing CSV files for multiple stocks", required=False)
    parser.add_argument("file", nargs="?", help="Path to a single CSV file for backtest", default=None)
    parser.add_argument("--tsl_trigger", type=float, default=10.0, help="Profit points after which TSL activates")
    parser.add_argument("--trail_size", type=float, default=5.0, help="Trail size in points once TSL is active")
    parser.add_argument("--debug", action="store_true", help="Show debug output for trade decisions (single file mode only)")
    parser.add_argument("--day", type=str, help="Date for debug mode, format YYYY-MM-DD")
    args = parser.parse_args()

    # Directory mode: previous code for per-year files, winrate filter, etc.
    if args.dir:
        # ... (Omitted for brevity, see previous version for per-year output logic)
        print("Debug mode is not valid in directory mode. Ignoring --debug and --day.")
        # You can add folder processing logic here (not covered in this snippet).
        return

    # Single file mode
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: {args.file} is not a valid file.")
            return

        df = load_data(args.file)
        pivots = calculate_pivots(df)

        # Debug mode
        if args.debug:
            if not args.day:
                print("You must provide --day with --debug. Example: --day 2022-03-24")
                return
            debug_trade_day(df, pivots, args.tsl_trigger, args.trail_size, args.day)
        else:
            trades_df = simulate_trades(df, pivots, args.tsl_trigger, args.trail_size)
            if trades_df.empty:
                print("No trades found for given data/year filter.")
                return
            print("\n=== Day-wise Trade Results ===")
            print(trades_df.to_string(index=False))
            # ... Optionally add summary output for single file mode here

    else:
        print("Error: Please provide either a single data file or use --dir for directory mode.")
        return

if __name__ == "__main__":
    main()

