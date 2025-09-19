import argparse
import pandas as pd
from datetime import time
from math import sqrt
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest strategy on 5m stock data or folder of stocks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv_file", help="Single CSV file with 5m OHLCV data")
    group.add_argument("--dir", help="Directory containing multiple CSV files")
    parser.add_argument("--profit_target", type=float, default=0.2,
                        help="Profit target as percentage (e.g. 0.2 for 0.2%)")
    parser.add_argument("--year", type=int, help="Year to backtest (if not provided with --dir, all years processed)")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logs")
    return parser.parse_args()

def strip_timezone(dt):
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        return dt.tz_localize(None) if hasattr(dt, 'tz_localize') else dt.replace(tzinfo=None)
    return dt

def run_backtest(df, year, stock_name, profit_target, debug):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].dt.tz_localize(None)
    df["date_only"] = df["Date"].dt.date
    df["time_only"] = df["Date"].dt.time

    # Filter data by year
    df_year = df[df["Date"].dt.year == year]
    if df_year.empty:
        if debug:
            print(f"No data for year {year} in {stock_name}")
        return None

    df_year.sort_values(by="Date", inplace=True)

    total_trades = 0
    total_long = 0
    total_short = 0
    wins = 0
    losses = 0
    no_trade_days = 0
    net_points = 0

    unique_dates = sorted(df_year["date_only"].unique())

    print(f"\nBacktest for stock: {stock_name} year: {year}\n")
    print(f"{'Date':10} | {'Trade':6} | {'Entry_Time':19} | {'Entry_Price':11} | {'Exit_Time':19} | {'Exit_Price':11} | {'Result':5} | {'P/L_Points':10}")
    print("-" * 110)

    for i in range(1, len(unique_dates)):
        day = unique_dates[i]
        prev_day = unique_dates[i-1]

        day_data = df_year[df_year["date_only"] == day]
        prev_day_data = df_year[df_year["date_only"] == prev_day]

        day_open = day_data.iloc[0]["Open"]
        prev_day_close = prev_day_data.iloc[-1]["Close"]
        tolerance_passed = abs(day_open - prev_day_close) <= 0.2

        if not tolerance_passed:
            no_trade_days += 1
            if debug:
                print(f"\n{day} - No trade: daily open {day_open:.4f} differs from previous close {prev_day_close:.4f} beyond tolerance 0.2")
            print(f"{day} | No Trade | -                 | -           | -                 | -           | -     | -         ")
            continue

        first_hour_candles = day_data.iloc[:12]
        first_hour_high = first_hour_candles["High"].max()
        first_hour_low = first_hour_candles["Low"].min()

        dd_long = sqrt(first_hour_high) * 0.2611
        dd_short = sqrt(first_hour_low) * 0.2611

        long_entry_level = first_hour_high + dd_long
        long_sl = first_hour_high - dd_long
        short_entry_level = first_hour_low - dd_short
        short_sl = first_hour_low + dd_short

        if debug:
            print(f"\n{day} - First hour high: {first_hour_high:.4f}, low: {first_hour_low:.4f}")
            print(f"{day} - Long DD: {dd_long:.4f}, Short DD: {dd_short:.4f}")
            print(f"{day} - Long entry level: {long_entry_level:.4f}, SL: {long_sl:.4f}")
            print(f"{day} - Short entry level: {short_entry_level:.4f}, SL: {short_sl:.4f}")

        entry_window = day_data[
            (day_data["time_only"] >= time(10,20)) &
            (day_data["time_only"] <= time(14,50))
        ].copy()

        if entry_window.empty:
            no_trade_days += 1
            if debug:
                print(f"\n{day} - No trade: No candles in entry window (10:20 to 14:50)")
            print(f"{day} | No Trade | -                 | -           | -                 | -           | -     | -         ")
            continue

        trade_entered = False
        trade_type = None
        entry_price = None
        entry_time = None
        profit_target_price = None
        stop_loss_price = None
        trade_exit_price = None
        trade_exit_time = None

        for idx, row in entry_window.iterrows():
            close_price = row["Close"]
            candle_time = row["Date"]

            if not trade_entered:
                if close_price > long_entry_level:
                    trade_type = "Long"
                    entry_price = close_price
                    entry_time = candle_time
                    profit_target_price = max(entry_price * (1 + profit_target / 100), entry_price + 10)
                    stop_loss_price = long_sl
                    total_trades += 1
                    total_long += 1
                    trade_entered = True
                    if debug:
                        print(f"{day} - Long entry at {entry_time} price {entry_price:.4f} (Target: {profit_target_price:.4f}, SL: {stop_loss_price:.4f})")
                    break
                elif close_price < short_entry_level:
                    trade_type = "Short"
                    entry_price = close_price
                    entry_time = candle_time
                    profit_target_price = min(entry_price * (1 - profit_target / 100), entry_price - 10)
                    stop_loss_price = short_sl
                    total_trades += 1
                    total_short += 1
                    trade_entered = True
                    if debug:
                        print(f"{day} - Short entry at {entry_time} price {entry_price:.4f} (Target: {profit_target_price:.4f}, SL: {stop_loss_price:.4f})")
                    break

        if not trade_entered:
            no_trade_days += 1
            if debug:
                print(f"\n{day} - No trade: No entry signal triggered between 10:20 and 14:50")
            print(f"{day} | No Trade | -                 | -           | -                 | -           | -     | -         ")
            continue

        post_entry_data = day_data[day_data["Date"] >= entry_time]

        exited = False
        for _, candle in post_entry_data.iterrows():
            high = candle["High"]
            low = candle["Low"]
            candle_time = candle["Date"]

            if trade_type == "Long":
                if low <= stop_loss_price:
                    trade_exit_price = stop_loss_price
                    trade_exit_time = candle_time
                    losses += 1
                    net_points += (trade_exit_price - entry_price)
                    exited = True
                    if debug:
                        print(f"{day} - Long SL hit at {trade_exit_time} price {trade_exit_price:.4f}")
                    break
                if high >= profit_target_price:
                    trade_exit_price = profit_target_price
                    trade_exit_time = candle_time
                    wins += 1
                    net_points += (trade_exit_price - entry_price)
                    exited = True
                    if debug:
                        print(f"{day} - Long Target hit at {trade_exit_time} price {trade_exit_price:.4f}")
                    break

            elif trade_type == "Short":
                if high >= stop_loss_price:
                    trade_exit_price = stop_loss_price
                    trade_exit_time = candle_time
                    losses += 1
                    net_points += (entry_price - trade_exit_price)
                    exited = True
                    if debug:
                        print(f"{day} - Short SL hit at {trade_exit_time} price {trade_exit_price:.4f}")
                    break
                if low <= profit_target_price:
                    trade_exit_price = profit_target_price
                    trade_exit_time = candle_time
                    wins += 1
                    net_points += (entry_price - trade_exit_price)
                    exited = True
                    if debug:
                        print(f"{day} - Short Target hit at {trade_exit_time} price {trade_exit_price:.4f}")
                    break

        if not exited:
            day_close_price = day_data.iloc[-1]["Close"]
            trade_exit_price = day_close_price
            trade_exit_time = day_data.iloc[-1]["Date"]

            if trade_type == "Long":
                pnl = trade_exit_price - entry_price
            else:
                pnl = entry_price - trade_exit_price

            net_points += pnl
            
            if pnl > 0:
                wins += 1
                result = "Win"
                if debug:
                    print(f"{day} - Exit at close price {trade_exit_price:.4f} (Win) at {trade_exit_time}")
            else:
                losses += 1
                result = "Loss"
                if debug:
                    print(f"{day} - Exit at close price {trade_exit_price:.4f} (Loss) at {trade_exit_time}")
        else:
            pnl = (trade_exit_price - entry_price) if trade_type == "Long" else (entry_price - trade_exit_price)
            result = "Win" if pnl > 0 else "Loss"

        entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M:%S")
        trade_exit_time_str = trade_exit_time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"{day} | {trade_type:6} | {entry_time_str} | {entry_price:11.4f} | {trade_exit_time_str} | {trade_exit_price:11.4f} | {result:5} | {pnl:10.4f}")

    print(f"\nSummary for {stock_name} {year}:")
    print(f"Total Trades   : {total_trades}")
    print(f"Total Longs    : {total_long}")
    print(f"Total Shorts   : {total_short}")
    print(f"Winning Trades : {wins}")
    print(f"Losing Trades  : {losses}")
    print(f"No Trade Days  : {no_trade_days}")
    print(f"Net Points     : {net_points:.4f}\n")

def main():
    args = parse_args()

    if args.csv_file:
        # Single file mode
        stock_name = os.path.basename(args.csv_file)
        df = pd.read_csv(args.csv_file)
        if args.year:
            years_to_process = [args.year]
        else:
            df["Date"] = pd.to_datetime(df["Date"])
            years_to_process = sorted(df["Date"].dt.year.unique())
        for year in years_to_process:
            run_backtest(df, year, stock_name, args.profit_target, args.debug)

    elif args.dir:
        # Folder mode
        csv_files = glob.glob(os.path.join(args.dir, "*.csv"))
        if not csv_files:
            print(f"No CSV files found in directory {args.dir}")
            return

        for csv_path in csv_files:
            stock_name = os.path.basename(csv_path)
            try:
                df = pd.read_csv(csv_path)
                df["Date"] = pd.to_datetime(df["Date"])
            except Exception as e:
                print(f"Skipping {stock_name} due to read error: {e}")
                continue

            if args.year:
                years_to_process = [args.year]
            else:
                years_to_process = sorted(df["Date"].dt.year.unique())

            for year in years_to_process:
                print(f"\nProcessing {stock_name}, Year: {year}")
                run_backtest(df, year, stock_name, args.profit_target, args.debug)

if __name__ == "__main__":
    main()

