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

def calc_max_drawdown(pnls, initial_equity=DEFAULT_START_EQUITY):
    equity = initial_equity + np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    min_drawdown = drawdowns.min()
    peak_equity = running_max.iloc[np.argmin(drawdowns)]
    dd_pct = (min_drawdown / peak_equity) * 100 if peak_equity != 0 else 0
    return min_drawdown, dd_pct

def summary_for_stock_year(stock_name, trades_df, year):
    if trades_df.empty:
        print(f"\n=== Stock: {stock_name}, Year: {year} ===")
        print("No trades.")
        return None

    total_pnl = trades_df['PnL'].sum()
    wins = (trades_df['PnL'] > 0).sum()
    losses = (trades_df['PnL'] < 0).sum()
    trades_count = len(trades_df)
    max_dd, max_dd_pct = calc_max_drawdown(trades_df['PnL'])
    win_rate = (wins / trades_count) * 100 if trades_count > 0 else 0.0

    print(f"\n=== Stock: {stock_name}, Year: {year} ===")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Number of Trades: {trades_count}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Max Drawdown: {max_dd:.2f}  ({max_dd_pct:.2f}%)")

    if win_rate < 70:
        return None  # Skip stocks with win rate below 70%

    return {
        "Stock": stock_name,
        "Year": year,
        "Total_PnL": total_pnl,
        "Trades": trades_count,
        "Wins": wins,
        "Losses": losses,
        "WinRate%": win_rate,
        "Max_Drawdown": max_dd,
        "MaxDD_%": max_dd_pct
    }

def backup_existing_file(filepath):
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{os.path.splitext(filepath)[0]}_backup_{timestamp}.csv"
        shutil.move(filepath, backup_name)
        print(f"Existing file backed up as {backup_name}")

def main():
    parser = argparse.ArgumentParser(description="Backtest Pivot Breakout Strategy with Per-Year CSV files and WinRate Filter")
    parser.add_argument("--dir", help="Directory containing CSV files for multiple stocks", required=False)
    parser.add_argument("file", nargs="?", help="Path to a single CSV file for backtest", default=None)
    parser.add_argument("--tsl_trigger", type=float, default=10.0, help="Profit points after which TSL activates")
    parser.add_argument("--trail_size", type=float, default=5.0, help="Trail size in points once TSL is active")
    args = parser.parse_args()

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: {args.dir} is not a valid directory.")
            return

        for year in range(2015, 2026):  # inclusive 2015-2025
            export_filename = f'195pivotallstocksperf_{year}.csv'
            backup_existing_file(export_filename)

            yearly_results = []

            print(f"\n### Running backtest for Year: {year} ###\n")
            for fname in sorted(os.listdir(args.dir)):
                if not fname.lower().endswith(".csv"):
                    continue
                fpath = os.path.join(args.dir, fname)
                stock_name = os.path.splitext(fname)[0]
                df = load_data(fpath, year)
                pivots = calculate_pivots(df)
                trades_df = simulate_trades(df, pivots, args.tsl_trigger, args.trail_size)
                result = summary_for_stock_year(stock_name, trades_df, year)
                if result is not None:
                    yearly_results.append(result)

            if yearly_results:
                yearly_df = pd.DataFrame(yearly_results)
                yearly_df = yearly_df[['Stock', 'Year', 'Total_PnL', 'Trades', 'Wins', 'Losses', 'WinRate%', 'Max_Drawdown', 'MaxDD_%']]
                yearly_df = yearly_df.sort_values(by='WinRate%', ascending=False)
                yearly_df.to_csv(export_filename, index=False)
                print(f"Saved year {year} performance to: {export_filename}")
            else:
                print(f"No stocks with WinRate >= 70% for year {year}. No file created.")

    elif args.file:
        df = load_data(args.file)
        pivots = calculate_pivots(df)
        trades_df = simulate_trades(df, pivots, args.tsl_trigger, args.trail_size)
        if trades_df.empty:
            print("No trades found for given data/year filter.")
            return
        print("\n=== Day-wise Trade Results ===")
        print(trades_df.to_string(index=False))
        summary_for_stock_year(os.path.splitext(os.path.basename(args.file))[0], trades_df, df['datetime'].dt.year.iloc)
    else:
        print("Error: Please provide either a single file or use --dir for directory mode.")
        return


if __name__ == "__main__":
    main()

