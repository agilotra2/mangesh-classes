import argparse
import os
import sys
import glob
import pandas as pd
from datetime import datetime

def print_rules():
    rules_text = """
Strategy Rules and Logic:

1. Entry Rules:
   - Long only strategy.
   - Consider three consecutive daily candles: Day[-2], Day[-1], Day[0].
   - Body of Day[-2] must be strictly within the body of Day[-1].
   - Body of Day[0] must be strictly within the body of Day[-1].
   - After this pattern, entry can happen on any of the next 5 trading days after Day[0].
   - Entry triggers on the first day the high crosses Day[0] high.
   - No entry if price falls below Day[0] low before entry within these 5 days.
   - Only one trade allowed per day.

2. Exit Rules:
   - Take Profit (TP): the higher of ₹20 absolute move or 0.2% of entry price.
   - Stop Loss (SL): low of the entry candle.
   - Exit when TP or SL is hit on any subsequent day or exit on last available data day.

3. Body (BO and BD) calculation:
   - Body is the range between the open and close prices of a candle.
   - Strictly within means the inner body's low > outer body's low, and inner body's high < outer body's high.

4. Multiple Trades or Stops:
   - The strategy allows only one trade per day.
   - Each trade managed independently with fixed SL and TP.

5. Trading Window:
   - Trades/signals are checked daily.
   - Signal for entry after pattern can trigger entry within 5 days respecting price conditions.
   - Days with missing OHLC are skipped.

Input CSV File Format:
- Columns required: Date, Open, High, Low, Close
- 'Date' column must be parseable as datetime.
- Data should be daily OHLC prices.
- Missing data rows are ignored.

Output Format:
- Displays a pretty summary table of trades on the console.
- Saves a CSV summary of all trades with timestamped filename.
- In directory mode (--dir), saves only one combined CSV summary of all files.
- Shows aggregated statistics like total trades, total PnL, win rate.
- Supports debug logs for detailed stepwise trade logic.

Usage Notes:
- Use --filepath for a single file, --dir for folder processing.
- Use --year to filter trades by calendar year.
- Use --debug to print detailed trading logic.
- Use --day with --debug to focus debug output on specific day.
- Use --showrules to print these rules and exit without backtesting.

"""
    print(rules_text)

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest price action long-only strategy.")
    parser.add_argument("filepath", nargs="?", help="Path to CSV file for single stock backtest.")
    parser.add_argument("--dir", help="Directory path to process all CSV files in folder.")
    parser.add_argument("--year", type=int, help="Filter trades by this year.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging.")
    parser.add_argument("--day", help="When debugging, show detailed decision for this day (YYYY-MM-DD).")
    parser.add_argument("--showrules", action="store_true", help="Show strategy rules and exit.")
    return parser.parse_args()

def get_body_range(row):
    return (min(row['Open'], row['Close']), max(row['Open'], row['Close']))

def is_body_within(inner, outer):
    return (inner[0] > outer[0]) and (inner[1] < outer[1])

def backtest_df(df, stock_name, year_filter, debug, debug_day):
    df = df.dropna(subset=['Open','High','Low','Close'])
    df = df.sort_values('Date').reset_index(drop=True)

    if year_filter:
        df['Year'] = df['Date'].dt.year
        df = df[df['Year'] == year_filter].reset_index(drop=True)
        if df.empty:
            if debug:
                print(f"[{stock_name}] No data for year {year_filter}.")
            return pd.DataFrame(), {}

    trades = []
    last_trade_date = None

    def debug_log(day, msg):
        if debug and (debug_day is None or debug_day == day):
            print(f"[{stock_name}][{day}] {msg}")

    for i in range(2, len(df)-1):
        day_m2 = df.loc[i-2]
        day_m1 = df.loc[i-1]
        day_0  = df.loc[i]

        daystr_0 = day_0['Date'].strftime("%Y-%m-%d")

        body_m2 = get_body_range(day_m2)
        body_m1 = get_body_range(day_m1)
        body_0  = get_body_range(day_0)

        debug_log(daystr_0, f"Body[-2]: {body_m2}, Body[-1]: {body_m1}, Body[0]: {body_0}")

        cond1 = is_body_within(body_m2, body_m1)
        cond2 = is_body_within(body_0, body_m1)

        debug_log(daystr_0, f"Body[-2] within Body[-1]: {cond1}, Body[0] within Body[-1]: {cond2}")

        if not (cond1 and cond2):
            debug_log(daystr_0, "Entry conditions not met.")
            continue

        day0_high = day_0['High']
        day0_low = day_0['Low']

        entry_triggered = False
        entry_price = None
        entry_day_str = None
        entry_index = None
        stop_loss = None

        for offset in range(1, 6):
            check_index = i + offset
            if check_index >= len(df):
                break
            day_check = df.loc[check_index]
            daystr_check = day_check['Date'].strftime("%Y-%m-%d")

            if last_trade_date == daystr_check:
                debug_log(daystr_check, "Skipping entry: already traded this day.")
                continue

            if day_check['Low'] < day0_low:
                debug_log(daystr_check, f"Entry invalidated: low {day_check['Low']:.2f} < Day[0] low {day0_low:.2f}.")
                break

            if day_check['High'] > day0_high and not entry_triggered:
                entry_triggered = True
                entry_price = day0_high
                entry_day_str = daystr_check
                entry_index = check_index
                stop_loss = day_check['Low']
                break

        if not entry_triggered:
            debug_log(daystr_0, "No entry triggered within next 5 days.")
            continue

        tp_abs = 20
        tp_perc = entry_price * 0.002
        take_profit = max(tp_abs, tp_perc)

        debug_log(entry_day_str, f"Entry triggered at {entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")

        exited = False
        exit_price = None
        exit_date = None
        exit_reason = None

        for j in range(entry_index, len(df)):
            candle = df.loc[j]
            daystr_exit = candle['Date'].strftime("%Y-%m-%d")

            if pd.isna(candle['High']) or pd.isna(candle['Low']):
                debug_log(daystr_exit, "Skipping exit day due to missing data.")
                continue

            tp_level = entry_price + take_profit
            if candle['High'] >= tp_level:
                exit_price = tp_level
                exit_date = daystr_exit
                exit_reason = 'TP'
                debug_log(daystr_exit, f"TP hit at {exit_price:.2f} on {exit_date}.")
                exited = True
                break

            if candle['Low'] <= stop_loss:
                exit_price = stop_loss
                exit_date = daystr_exit
                exit_reason = 'SL'
                debug_log(daystr_exit, f"SL hit at {exit_price:.2f} on {exit_date}.")
                exited = True
                break

        if not exited:
            last_candle = df.loc[len(df)-1]
            exit_price = last_candle['Close']
            exit_date = last_candle['Date'].strftime("%Y-%m-%d")
            exit_reason = 'EOD'
            debug_log(exit_date, f"Exit on EOD at {exit_price:.2f}.")

        pnl = exit_price - entry_price

        trades.append(dict(
            Entry_Date=entry_day_str,
            Entry_Price=entry_price,
            Exit_Date=exit_date,
            Exit_Price=exit_price,
            Exit_Reason=exit_reason,
            SL=stop_loss,
            TP=entry_price+take_profit,
            PnL=pnl
        ))

        last_trade_date = entry_day_str
        debug_log(exit_date, f"Trade closed. PnL={pnl:.2f}")

    if not trades:
        if debug:
            print(f"No trades found for {stock_name} year {year_filter or 'all'}.")
        return pd.DataFrame(), {}

    trades_df = pd.DataFrame(trades)

    if not args.dir:
        print(f"\nFinal Trades Summary for {stock_name} year {year_filter or 'all'}:")
        print(trades_df.to_string(index=False))
        print()

    total_trades = len(trades_df)
    total_pnl = trades_df['PnL'].sum()
    avg_pnl = trades_df['PnL'].mean()
    wins = trades_df[trades_df['PnL'] > 0].shape[0]
    losses = trades_df[trades_df['PnL'] <= 0].shape[0]
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    stats = dict(
        stock=stock_name,
        year=year_filter or 'all',
        total_trades=total_trades,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        wins=wins,
        losses=losses,
        win_rate=win_rate
    )

    return trades_df, stats

def save_csv(trades_df, script_name):
    nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{script_name}_summary_{nowstr}.csv"
    trades_df.to_csv(filename, index=False)
    print(f"Full trade summary saved to: {filename}")

def process_single_file(filepath, year, debug, debug_day):
    stock = os.path.splitext(os.path.basename(filepath))[0]
    df = pd.read_csv(filepath)
    if 'Date' not in df.columns:
        print(f"[{stock}] File missing 'Date' column. Skipping.")
        return {}
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        print(f"[{stock}] 'Date' column parse error: {e}. Skipping.")
        return {}
    trades_df, stats = backtest_df(df, stock, year, debug, debug_day)
    if not trades_df.empty:
        save_csv(trades_df, stock)
    return stats

def process_directory(dirpath, year, debug):
    all_stats = []
    all_trades = []
    csv_files = glob.glob(os.path.join(dirpath, "*.csv"))
    for path in csv_files:
        stock = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[{stock}] Read error: {e}. Skipping file.")
            continue
        if 'Date' not in df.columns:
            print(f"[{stock}] Missing 'Date' column. Skipping file.")
            continue
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print(f"[{stock}] 'Date' parse error: {e}. Skipping file.")
            continue

        trades_df, stats = backtest_df(df, stock, year, debug, None)
        if trades_df.empty:
            print(f"[{stock}] No trades found.")
        else:
            print(f"[{stock}] Total Trades: {stats['total_trades']}, Total PnL: {stats['total_pnl']:.2f}")
            all_stats.append(stats)
            trades_df['Stock'] = stock
            trades_df['Year'] = stats.get('year', 'all')
            all_trades.append(trades_df)

    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_df = summary_df.sort_values('total_pnl', ascending=False).reset_index(drop=True)
        print("\nCombined Summary (per stock per year):")
        print(summary_df.to_string(index=False))

        top3 = summary_df.head(3)
        print("\nTop 3 stocks by Total PnL:")
        print(top3[['stock', 'year', 'total_pnl']].to_string(index=False))

        combined_trades_df = pd.concat(all_trades, ignore_index=True)
        nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"strategy_backtest_combined_summary_{nowstr}.csv"
        combined_trades_df.to_csv(combined_filename, index=False)
        print(f"\nCombined CSV summary saved to: {combined_filename}")
    else:
        print("No trades found across files.")

if __name__ == "__main__":
    args = parse_args()

    if args.showrules:
        print_rules()
        sys.exit(0)

    if not args.filepath and not args.dir:
        print("ERROR: Provide either a CSV file path or --dir directory path to process.")
        sys.exit(1)

    if args.filepath:
        process_single_file(args.filepath, args.year, args.debug, args.day)

    if args.dir:
        process_directory(args.dir, args.year, args.debug)

