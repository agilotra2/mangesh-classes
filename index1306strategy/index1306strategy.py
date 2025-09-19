#!/usr/bin/env python3
import pandas as pd
import argparse
import os
from datetime import datetime, time

def print_rules():
    print("""
========================
Backtest Strategy Rules
========================

1. CSV File Format Required:
   - The file must have columns: date, open, high, low, close (case-insensitive).
   - 'date' column should contain timestamps in a format parsable by pandas, e.g. 2022-01-03 09:15:00.
   - Data should be 1-minute OHLC (Open, High, Low, Close) intraday bars.

2. How BO and BD are calculated:
   - The first 5-minute candle of the day is used as basis.
     > First5MinHigh = max(high of first 5 1-min bars)
     > First5MinLow = min(low of first 5 1-min bars)
   - BO (breakout) = First5MinHigh + First5MinHigh * 0.001306
   - BD (breakdown) = First5MinLow - First5MinLow * 0.001306

3. Trades are taken as follows:
   - Aggregate all intraday 1-min bars into 5-min bars.
   - Starting from the second 5-min candle (i.e, after 09:20 for a 09:15 open), 
     > If a 5-min bar closes above BO, enter LONG at that close.
     > If a 5-min bar closes below BD, enter SHORT at that close.
   - Only one trade per day: after the first trade (whether profit or loss), no more trades are attempted.
   - Entry, stop loss, and take profit are handled as follows:
     > LONG: Entry = 5-min close; TP = Entry+20, SL = Entry-20
     > SHORT: Entry = 5-min close; TP = Entry-20, SL = Entry+20
   - After entry, check each following 1-min bar:
     > If price reaches TP, trade is closed at TP, marked as "TP".
     > If price reaches SL, trade is closed at SL, marked as "SL".
     > If neither is reached by end of day, trade is marked "EOD" with zero PnL.
   - No more entries are considered after 14:00. Entries/signals after this time are ignored.

4. Output Provided:
   - Prints a summary table (date, trade type, entry/exit details, outcome, PnL for each day).
   - Saves complete summary (including all details above) as a CSV with a timestamp in the filename.
   - If --year is supplied, also prints one-line total PnL for the filtered year.

==================
""")
    print("No backtest was run (`--showrules` specified).")
    exit(0)

def debug_print(debug, day_filter, current_day, msg):
    if debug and (day_filter is None or day_filter == current_day):
        print(msg)

def get_5min_candles(df):
    df = df.copy()
    df['5min_period'] = (df['date'].dt.hour * 60 + df['date'].dt.minute - 9*60 - 15) // 5
    grouped = df.groupby('5min_period')
    ohlc = grouped.agg(open=('open','first'), high=('high','max'),
                       low=('low','min'), close=('close','last'), date=('date','last')).reset_index()
    return ohlc

def process_day(df, debug=False, day_filter=None):
    day = df['date'].dt.date.iloc[0]
    if day_filter and day != day_filter:
        return None

    debug_print(debug, day_filter, day, f"\n=== Processing {day} ===")

    if len(df) < 6:
        debug_print(debug, day_filter, day, f"[{day}] Skipped: Insufficient data (<6 minutes)")
        return {
            "date": day, "trade": "Skipped", "entry_time": None,
            "entry_side": None, "entry_price": None, "exit_time": None,
            "exit_price": None, "exit_reason": None, "PnL": 0.0
        }

    first5 = df.iloc[:5]
    first5_high = first5['high'].max()
    first5_low = first5['low'].min()
    first5_open = first5['open'].iloc[0]

    BO = first5_high + first5_high * 0.001306
    BD = first5_low - first5_low * 0.001306

    debug_print(debug, day_filter, day,
        f"First 5-min candle Open={first5_open}, High={first5_high}, Low={first5_low}")
    debug_print(debug, day_filter, day,
        f"BO: {BO:.4f} | BD: {BD:.4f} | SL fixed (20 points R:R=1:1 after entry as per previous logic)")

    five_min_candles = get_5min_candles(df)

    trade_taken = False
    entry_time = None
    entry_price = None
    entry_side = None
    TP = None
    SL_level = None

    # Check from second 5-min candle onwards for close above BO or below BD
    for idx in range(1, len(five_min_candles)):
        candle = five_min_candles.iloc[idx]
        cand_close = candle['close']
        cand_time = candle['date']
        if cand_time.time() >= time(14,0):
            break

        if debug and (day_filter is None or day_filter == day):
            debug_print(debug, day_filter, day,
                f"5-min Candle #{idx+1} Close: {cand_close:.4f} at {cand_time.strftime('%H:%M:%S')}")

        if not trade_taken:
            if cand_close > BO:
                entry_side = "LONG"
                entry_time = cand_time
                entry_price = cand_close
                TP = entry_price + 20
                SL_level = entry_price - 20
                trade_taken = True
                debug_print(debug, day_filter, day,
                    f"LONG entry at {entry_time} price={entry_price:.4f} TP={TP:.4f} SL={SL_level:.4f}")
                break
            elif cand_close < BD:
                entry_side = "SHORT"
                entry_time = cand_time
                entry_price = cand_close
                TP = entry_price - 20
                SL_level = entry_price + 20
                trade_taken = True
                debug_print(debug, day_filter, day,
                    f"SHORT entry at {entry_time} price={entry_price:.4f} TP={TP:.4f} SL={SL_level:.4f}")
                break

    if not trade_taken:
        debug_print(debug, day_filter, day, "No trade triggered this day.")
        return {
            "date": day, "trade": "No setup", "entry_time": None,
            "entry_side": None, "entry_price": None, "exit_time": None,
            "exit_price": None, "exit_reason": None, "PnL": 0.0
        }

    # Find the index in 1-min data where entry_time is
    entry_idx = df.index[df['date'] == entry_time]
    if len(entry_idx) == 0:
        entry_idx = df.index[df['date'] >= entry_time]
    entry_idx = entry_idx[0] if len(entry_idx) > 0 else None
    if entry_idx is None:
        debug_print(debug, day_filter, day, "Entry time not found in 1-min data, no exit simulation possible.")
        return {
            "date": day, "trade": entry_side, "entry_time": entry_time,
            "entry_side": entry_side, "entry_price": entry_price,
            "exit_time": None, "exit_price": None,
            "exit_reason": "EOD", "PnL": 0.0
        }

    for i in range(entry_idx+1, len(df)):
        row = df.iloc[i]
        price = row['close']
        time_ = row['date']
        if time_.time() >= time(14,0):
            break

        if entry_side == "LONG":
            if price >= TP:
                debug_print(debug, day_filter, day, f"LONG exit at TP {TP:.4f} on {time_}")
                pnl = TP - entry_price
                return {
                    "date": day, "trade": "LONG", "entry_time": entry_time, "entry_side": entry_side,
                    "entry_price": entry_price, "exit_time": time_, "exit_price": TP,
                    "exit_reason": "TP", "PnL": pnl
                }
            elif price <= SL_level:
                debug_print(debug, day_filter, day, f"LONG exit at SL {SL_level:.4f} on {time_}")
                pnl = SL_level - entry_price
                return {
                    "date": day, "trade": "LONG", "entry_time": entry_time, "entry_side": entry_side,
                    "entry_price": entry_price, "exit_time": time_, "exit_price": SL_level,
                    "exit_reason": "SL", "PnL": pnl
                }
        else:
            if price <= TP:
                debug_print(debug, day_filter, day, f"SHORT exit at TP {TP:.4f} on {time_}")
                pnl = entry_price - TP
                return {
                    "date": day, "trade": "SHORT", "entry_time": entry_time, "entry_side": entry_side,
                    "entry_price": entry_price, "exit_time": time_, "exit_price": TP,
                    "exit_reason": "TP", "PnL": pnl
                }
            elif price >= SL_level:
                debug_print(debug, day_filter, day, f"SHORT exit at SL {SL_level:.4f} on {time_}")
                pnl = entry_price - SL_level
                return {
                    "date": day, "trade": "SHORT", "entry_time": entry_time, "entry_side": entry_side,
                    "entry_price": entry_price, "exit_time": time_, "exit_price": SL_level,
                    "exit_reason": "SL", "PnL": pnl
                }

    debug_print(debug, day_filter, day, "Position closed at End of Day with no TP/SL hit")
    return {
        "date": day, "trade": entry_side, "entry_time": entry_time, "entry_side": entry_side,
        "entry_price": entry_price, "exit_time": None, "exit_price": None,
        "exit_reason": "EOD", "PnL": 0.0
    }

def pretty_print_summary(results):
    results = [r for r in results if r is not None]
    if not results:
        print("No results to show.")
        return
    header = (f"{'Date':<12} | {'Trade':<10} | {'Entry Time':<20} | {'Side':<6} | "
              f"{'Entry Price':<12} | {'Exit Time':<20} | {'Exit Price':<12} | {'Reason':<6} | {'PnL':<12}")
    print(header)
    print("-"*len(header))
    for r in results:
        date_str = r['date'].strftime("%Y-%m-%d")
        entry_time = r['entry_time'].strftime("%H:%M:%S") if r['entry_time'] else "-"
        exit_time = r['exit_time'].strftime("%H:%M:%S") if r['exit_time'] else "-"
        trade = r['trade']
        side = r.get('entry_side', '-') if r.get('entry_side') else '-'
        entry_price = f"{r['entry_price']:.4f}" if r['entry_price'] else "-"
        exit_price = f"{r['exit_price']:.4f}" if r['exit_price'] else "-"
        reason = r['exit_reason'] if r['exit_reason'] else "-"
        pnl = f"{r['PnL']:.4f}"
        print(f"{date_str:<12} | {trade:<10} | {entry_time:<20} | {side:<6} | {entry_price:<12} | "
              f"{exit_time:<20} | {exit_price:<12} | {reason:<6} | {pnl:<12}")

def backtest(filepath, debug=False, day_filter=None, year_filter=None):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.sort_values('date', inplace=True)

    if year_filter is not None:
        df = df[df['date'].dt.year == year_filter]
        if df.empty:
            print(f"No data found for year {year_filter}")
            return

    final_results = []
    days_processed = df['date'].dt.date.unique()

    for day in days_processed:
        day_df = df[df['date'].dt.date == day].reset_index(drop=True)
        result = process_day(day_df, debug=debug, day_filter=day_filter)
        if result is not None:
            final_results.append(result)

    pretty_print_summary(final_results)

    if year_filter is not None:
        total_pnl = sum(r.get("PnL", 0.0) for r in final_results if r is not None)
        print(f"\n==== FINAL PnL SUMMARY for Year {year_filter}: {total_pnl:.4f} points ====")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"{script_name}_summary_{timestamp}.csv"
    pd.DataFrame(final_results).to_csv(out_filename, index=False)
    print(f"\nSummary saved to {out_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest 1-min index strategy: entry on 5-min candle close BO/BD.")
    parser.add_argument('csvfile', nargs='?', help="Input CSV file with 1-min OHLC data")
    parser.add_argument('--showrules', action='store_true', help="Show all implemented strategy rules and required file format")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")
    parser.add_argument('--day', type=str, default=None, help="Run backtest for specific day (YYYY-MM-DD) with debug")
    parser.add_argument('--year', type=int, default=None, help="Run backtest for specific year")

    args = parser.parse_args()

    if args.showrules:
        print_rules()

    day_filter = None
    if args.day:
        try:
            day_filter = datetime.strptime(args.day, "%Y-%m-%d").date()
        except ValueError:
            print("Error: --day argument must be in YYYY-MM-DD format.")
            exit(1)

    if not args.csvfile:
        print("Error: CSV input file required unless using --showrules.")
        exit(1)

    backtest(args.csvfile, debug=args.debug, day_filter=day_filter, year_filter=args.year)

