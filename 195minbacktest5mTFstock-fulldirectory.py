import argparse
import pandas as pd
import numpy as np
import os
from datetime import time

DEFAULT_START_EQUITY = 100000

def load_data(file_path, year=None):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.rename(columns={'Date': 'datetime',
                       'Open': 'open', 'High': 'high',
                       'Low': 'low', 'Close': 'close',
                       'Volume': 'volume'}, inplace=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    if year:
        df = df[df['datetime'].dt.year == int(year)]
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
    """
    pnls: series or list of daily (or trade-by-trade) profit/loss values
    Returns (max_drawdown_abs, max_drawdown_pct)
    """
    equity = initial_equity + np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    min_drawdown = drawdowns.min()
#    peak_equity = running_max[np.argmin(drawdowns)]
    peak_equity = running_max.iloc[np.argmin(drawdowns)]
    # Handle no trades (avoid division by zero)
    if peak_equity == 0:
        dd_pct = 0
    else:
        dd_pct = (min_drawdown / peak_equity) * 100
    return min_drawdown, dd_pct

def summary_for_stock(stock_name, trades_df, year_filter):
    total_pnl = trades_df['PnL'].sum()
    wins, losses, trades_count = (trades_df['PnL'] > 0).sum(), (trades_df['PnL'] < 0).sum(), len(trades_df)
    max_dd, max_dd_pct = calc_max_drawdown(trades_df['PnL'])

    print(f"\n=== Stock: {stock_name} ===")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Number of Trades: {trades_count}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {((wins / trades_count) * 100 if trades_count > 0 else 0):.2f}%")
    print(f"Max Drawdown: {max_dd:.2f}  ({max_dd_pct:.2f}%)")

    # Per-year breakdown if no year filter
    if not year_filter:
        print("--- Per-Year Summary ---")
        yearly_stats = trades_df.groupby("Year").agg(
            Total_PnL=("PnL", "sum"),
            Trades=("PnL", "count"),
            Wins=("PnL", lambda x: (x > 0).sum()),
            Losses=("PnL", lambda x: (x < 0).sum())
        ).reset_index()
        yearly_max_dd = []
        yearly_max_dd_pct = []
        for y in yearly_stats["Year"]:
            sub = trades_df[trades_df["Year"] == y]
            dd, dd_pct = calc_max_drawdown(sub['PnL'])
            yearly_max_dd.append(dd)
            yearly_max_dd_pct.append(dd_pct)
        yearly_stats["Max_Drawdown"] = [f"{dd:.2f}" for dd in yearly_max_dd]
        yearly_stats["MaxDD_%"] = [f"{p:.2f}" for p in yearly_max_dd_pct]
        yearly_stats["WinRate%"] = (yearly_stats["Wins"] / yearly_stats["Trades"]) * 100
        print(yearly_stats.sort_values("Year").to_string(index=False, formatters={
            "Total_PnL": "{:.2f}".format,
            "WinRate%": "{:.2f}".format
        }))

    return stock_name, total_pnl, trades_count, wins, losses, max_dd, max_dd_pct

def main():
    parser = argparse.ArgumentParser(description="Backtest Pivot Breakout Strategy (Single file or Directory mode, with max drawdown)")
    parser.add_argument("--dir", help="Directory containing CSV files for multiple stocks", required=False)
    parser.add_argument("file", nargs="?", help="Path to a single CSV file for backtest", default=None)
    parser.add_argument("--year", help="Year filter for backtest", required=False)
    parser.add_argument("--tsl_trigger", type=float, default=10.0, help="Profit points after which TSL activates")
    parser.add_argument("--trail_size", type=float, default=5.0, help="Trail size in points once TSL is active")
    args = parser.parse_args()

    year_filter = args.year

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: {args.dir} is not a valid directory.")
            return
        combined_results = []
        for fname in sorted(os.listdir(args.dir)):
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(args.dir, fname)
            stock_name = os.path.splitext(fname)[0]
            df = load_data(fpath, year_filter)
            pivots = calculate_pivots(df)
            trades_df = simulate_trades(df, pivots, args.tsl_trigger, args.trail_size)
            if trades_df.empty:
                print(f"\n=== Stock: {stock_name} ===")
                print("No trades.")
                continue
            stock_summary = summary_for_stock(stock_name, trades_df, year_filter)
            combined_results.append(stock_summary)

        if combined_results:
            comb_df = pd.DataFrame(combined_results, columns=["Stock", "Total_PnL", "Trades", "Wins", "Losses", "Max_DD", "MaxDD_pct"])
            comb_df["WinRate%"] = (comb_df["Wins"] / comb_df["Trades"] * 100).round(2)
            comb_df = comb_df.sort_values("Total_PnL", ascending=False)
            print("\n=== Combined Summary for Directory ===")
            print(comb_df.drop(['Max_DD', 'MaxDD_pct'], axis=1).to_string(index=False, formatters={"Total_PnL": "{:.2f}".format}))
            totals = pd.DataFrame({
                "Stock": ["TOTAL"],
                "Total_PnL": [comb_df["Total_PnL"].sum()],
                "Trades": [comb_df["Trades"].sum()],
                "Wins": [comb_df["Wins"].sum()],
                "Losses": [comb_df["Losses"].sum()],
                "WinRate%": [(comb_df["Wins"].sum() / comb_df["Trades"].sum() * 100) if comb_df["Trades"].sum() > 0 else 0]
            })
            print(totals.to_string(index=False, formatters={"Total_PnL": "{:.2f}".format, "WinRate%": "{:.2f}".format}))

    elif args.file:
        if not os.path.isfile(args.file):
            print(f"Error: {args.file} is not a valid file.")
            return
        df = load_data(args.file, year_filter)
        pivots = calculate_pivots(df)
        trades_df = simulate_trades(df, pivots, args.tsl_trigger, args.trail_size)
        if trades_df.empty:
            print("No trades found for given data/year filter.")
            return
        if year_filter:
            print("\n=== Day-wise Trade Results ===")
            print(trades_df.to_string(index=False))
        summary_for_stock(os.path.splitext(os.path.basename(args.file))[0], trades_df, year_filter)
    else:
        print("Error: Please provide either a single file or use --dir for directory mode.")
        return


if __name__ == "__main__":
    main()

