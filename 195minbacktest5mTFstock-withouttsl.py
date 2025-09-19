import argparse
import pandas as pd
import numpy as np
import os
import shutil
from datetime import time, datetime

DEFAULT_START_EQUITY = 100000

def print_strategy_rules():
    rules = """
Strategy Rules:
---------------
- Entry Condition:
  * LONG trade triggered when price crosses above (previous day's second half high + DD_long).
  * SHORT trade triggered when price crosses below (previous day's second half low - DD_short).

- Stop-loss (SL):
  * LONG: SL set at entry candle low.
  * SHORT: SL set at (entry candle low + DD_short).

- Profit Target:
  * Fixed profit target set as a percentage of entry price (default 0.1%).
  * Profit target price = Entry price ± (Entry price * profit_target_pct / 100)
    - For LONG: target = entry price + target amount
    - For SHORT: target = entry price - target amount

- No trailing stops currently applied (TSL parameters exist but inactive).

- Exit:
  * Trade exits immediately at stop-loss or profit target hit, whichever occurs first.

- Filtering & Output:
  * Stocks with win rate < 70% per year are excluded from results files.
  * CSV output files are created per year with names like: 195pivotallstocksperf_YYYY.csv.
  * Results per year are sorted descending by Win Rate%.

- Debug option available to print detailed step-by-step logic for a given day (--debug and --day).

"""
    print(rules)

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

def simulate_trades(df, pivots, profit_target_pct):
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
        entry_type, entry_price, stop_loss, target_price = None, None, None, None
        entry_time, exit_time = None, None
        pnl = 0

        for _, row in day_data.iterrows():
            candle_time = row['datetime'].time()
            close_price = row['close']

            if candle_time > time(15, 5) and not trade_taken:
                break

            if not trade_taken:
                if close_price > high_level:
                    entry_type = "LONG"
                    entry_price = round(close_price, 2)
                    # Updated SL: entry candle low for LONG
                    stop_loss = round(row['low'], 2)
                    profit_target_amount = entry_price * profit_target_pct / 100
                    target_price = round(entry_price + profit_target_amount, 2)
                    trade_taken, entry_time = True, row['datetime']
                    continue
                elif close_price < low_level:
                    entry_type = "SHORT"
                    entry_price = round(close_price, 2)
                    stop_loss = round(row['low'] + dd_short, 2)
                    profit_target_amount = entry_price * profit_target_pct / 100
                    target_price = round(entry_price - profit_target_amount, 2)
                    trade_taken, entry_time = True, row['datetime']
                    continue

            elif trade_taken:
                if entry_type == "LONG":
                    if row['low'] <= stop_loss:
                        pnl = stop_loss - entry_price
                        exit_time = row['datetime']
                        exit_reason = "StopLoss"
                        break
                    elif row['high'] >= target_price:
                        pnl = target_price - entry_price
                        exit_time = row['datetime']
                        exit_reason = "Target"
                        break
                elif entry_type == "SHORT":
                    if row['high'] >= stop_loss:
                        pnl = entry_price - stop_loss
                        exit_time = row['datetime']
                        exit_reason = "StopLoss"
                        break
                    elif row['low'] <= target_price:
                        pnl = entry_price - target_price
                        exit_time = row['datetime']
                        exit_reason = "Target"
                        break

        if trade_taken:
            if exit_time is None:
                last_close = round(day_data.iloc[-1]['close'], 2)
                pnl = last_close - entry_price if entry_type == "LONG" else entry_price - last_close
                exit_time_str = "EOD"
                exit_reason = "EOD"
            else:
                exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M")

            results.append({
                "Date": curr_day,
                "Year": curr_day.year,
                "Type": entry_type,
                "Entry Time": entry_time.strftime("%Y-%m-%d %H:%M"),
                "Entry Price": entry_price,
                "Stop Loss": stop_loss,
                "Target": target_price,
                "Exit Time": exit_time_str,
                "Exit Reason": exit_reason,
                "PnL": round(pnl, 2),
            })

    return pd.DataFrame(results)

def debug_trade_day(df, pivots, profit_target_pct, debug_day):
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
    entry_type, entry_price, stop_loss, target_price = None, None, None, None
    entry_time, exit_time = None, None
    pnl = 0

    for _, row in day_data.iterrows():
        candle_time = row['datetime'].time()
        close_price = row['close']
        debug_line = f"{row['datetime']} | Price: {close_price:.2f}"

        if not trade_taken and candle_time > time(15, 5):
            print(debug_line + " | No trade taken before cutoff.")
            break

        if not trade_taken:
            debug_line += " | Checking entry..."
            if close_price > high_level:
                entry_type = "LONG"
                entry_price = round(close_price, 2)
                stop_loss = round(row['low'], 2)
                profit_target_amount = entry_price * profit_target_pct / 100
                target_price = round(entry_price + profit_target_amount, 2)
                trade_taken, entry_time = True, row['datetime']
                debug_line += f" | LONG ENTRY: Entry={entry_price}, SL={stop_loss}, Target={target_price}, Profit Target %={profit_target_pct}"
            elif close_price < low_level:
                entry_type = "SHORT"
                entry_price = round(close_price, 2)
                stop_loss = round(row['low'] + dd_short, 2)
                profit_target_amount = entry_price * profit_target_pct / 100
                target_price = round(entry_price - profit_target_amount, 2)
                trade_taken, entry_time = True, row['datetime']
                debug_line += f" | SHORT ENTRY: Entry={entry_price}, SL={stop_loss}, Target={target_price}, Profit Target %={profit_target_pct}"
            else:
                debug_line += " | No entry."
            print(debug_line)
            continue

        debug_line += f" | In Trade ({entry_type}) Entry={entry_price}, SL={stop_loss}, Target={target_price}"
        if entry_type == "LONG":
            if row['low'] <= stop_loss:
                pnl = stop_loss - entry_price
                exit_time = row['datetime']
                debug_line += f" | SL HIT, PnL: {pnl:.2f}"
                print(debug_line)
                break
            elif row['high'] >= target_price:
                pnl = target_price - entry_price
                exit_time = row['datetime']
                debug_line += f" | TARGET HIT, PnL: {pnl:.2f}"
                print(debug_line)
                break
        elif entry_type == "SHORT":
            if row['high'] >= stop_loss:
                pnl = entry_price - stop_loss
                exit_time = row['datetime']
                debug_line += f" | SL HIT, PnL: {pnl:.2f}"
                print(debug_line)
                break
            elif row['low'] <= target_price:
                pnl = entry_price - target_price
                exit_time = row['datetime']
                debug_line += f" | TARGET HIT, PnL: {pnl:.2f}"
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

    print(f"\n=== Stock: {stock_name}, Year: {year} Summary ===")
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
    parser = argparse.ArgumentParser(description="Backtest Pivot Breakout Strategy with updated SL, customizable profit target, year filter, WinRate Filter, Debugging, Summary, and ShowRules")
    parser.add_argument("--dir", help="Directory containing CSV files for multiple stocks", required=False)
    parser.add_argument("file", nargs="?", help="Path to a single CSV file for backtest", default=None)
    parser.add_argument("--year", help="Year filter for backtest", required=False)
    parser.add_argument("--tsl_trigger", type=float, default=10.0, help="Trailing stop loss trigger (unused currently)")
    parser.add_argument("--trail_size", type=float, default=5.0, help="Trailing stop loss size (unused currently)")
    parser.add_argument("--profit_target_pct", type=float, default=0.1, help="Profit target percentage of entry price (default 0.1%)")
    parser.add_argument("--debug", action="store_true", help="Show debug output for trade decisions (single file mode only)")
    parser.add_argument("--day", type=str, help="Date for debug mode, format YYYY-MM-DD")
    parser.add_argument("--showrules", action="store_true", help="Show strategy rules and exit")
    args = parser.parse_args()

    if args.showrules:
        print_strategy_rules()
        return

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: {args.dir} is not a valid directory.")
            return

        years_to_run = []
        if args.year:
            try:
                y = int(args.year)
                years_to_run = [y]
            except:
                print("Invalid year format for --year.")
                return
        else:
            years_to_run = list(range(2015, 2026))

        for year in years_to_run:
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
                trades_df = simulate_trades(df, pivots, args.profit_target_pct)
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
        if not os.path.isfile(args.file):
            print(f"Error: {args.file} is not a valid file.")
            return
        df = load_data(args.file, args.year)
        pivots = calculate_pivots(df)

        if args.debug:
            if not args.day:
                print("You must provide --day with --debug. Example: --day 2022-03-24")
                return
            debug_trade_day(df, pivots, args.profit_target_pct, args.day)
        else:
            trades_df = simulate_trades(df, pivots, args.profit_target_pct)
            if trades_df.empty:
                print("No trades found for given data/year filter.")
                return
            print("\n=== Day-wise Trade Results ===")
            print(trades_df.to_string(index=False))
            summary_for_stock_year(os.path.splitext(os.path.basename(args.file))[0], trades_df, args.year if args.year else trades_df['Year'].iloc)

    else:
        print("Error: Please provide either a single data file or use --dir for directory mode.")
        return

if __name__ == "__main__":
    main()

