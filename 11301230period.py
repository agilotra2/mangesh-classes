import os
import pandas as pd
import numpy as np
import argparse
from glob import glob
from tabulate import tabulate

def parse_cli():
    parser = argparse.ArgumentParser(description='Backtest intraday breakout strategy on 5m OHLCV data.')
    parser.add_argument('--file', type=str, help='Single CSV file to process')
    parser.add_argument('--dir', type=str, help='Directory of CSV files to process')
    parser.add_argument('--year', type=int, help='Year to filter (optional)')
    parser.add_argument('--profit_target_mode', choices=['max'], default='max',
                        help='Profit target mode: always use max(10, 0.2%% of entry price)')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug output')
    args = parser.parse_args()
    return args

def debug_print(args, *msgs):
    if args.debug:
        print("[DEBUG]", *msgs)

def load_data(filename, year=None):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    start_time = pd.to_datetime('09:15', format='%H:%M').time()
    end_time = pd.to_datetime('15:25', format='%H:%M').time()
    df = df[(df['Date'].dt.time >= start_time) & (df['Date'].dt.time <= end_time)]
    if year:
        df = df[df['Date'].dt.year == year]
    return df

def process_day(daydf, args):
    tz = daydf['Date'].dt.tz
    day_str = daydf['Date'].dt.date.iloc[0].strftime('%Y-%m-%d')
    window_start = pd.Timestamp(f'{day_str} 11:30:00').tz_localize(tz)
    window_end = pd.Timestamp(f'{day_str} 12:30:00').tz_localize(tz)
    last_entry_time = pd.Timestamp(f'{day_str} 14:50:00').tz_localize(tz)

    debug_print(args, f"Processing day: {day_str}")

    window = daydf[(daydf['Date'] >= window_start) & (daydf['Date'] < window_end)]
    if window.empty:
        debug_print(args, "No data in 11:30-12:30 window, skipping day")
        return None
    
    high = window['High'].max()
    low = window['Low'].min()
    dd_long = np.sqrt(high) * 0.2611
    dd_short = np.sqrt(low) * 0.2611

    debug_print(args, f"Window High: {high}, Low: {low}")
    debug_print(args, f"DD Long: {dd_long:.4f}, DD Short: {dd_short:.4f}")

    after = daydf[daydf['Date'] >= window_end]
    entry = None
    direction = None

    for idx, row in after.iterrows():
        close = row['Close']
        if row['Date'] > last_entry_time:
            debug_print(args, f"Current candle time {row['Date']} exceeds 14:50 - no more entry")
            break  # No entry after 14:50
        if close > high + dd_long:
            direction = 'Long'
            entry = row
            debug_print(args, f"Entry LONG at close={close} on {row['Date']}")
            break
        elif close < low - dd_short:
            direction = 'Short'
            entry = row
            debug_print(args, f"Entry SHORT at close={close} on {row['Date']}")
            break
    
    if entry is None or direction is None:
        debug_print(args, "No breakout entry signal found after 12:30")
        return None
    
    entry_price = entry['Close']
    entry_time = entry['Date']
    
    if direction == 'Long':
        sl_price = high - dd_long
        pt_points = max(10, entry_price * 0.002)
        pt_price = entry_price + pt_points
    else:
        sl_price = low + dd_short
        pt_points = max(10, entry_price * 0.002)
        pt_price = entry_price - pt_points

    debug_print(args, f"SL price: {sl_price:.4f}, PT price: {pt_price:.4f}")

    trade_outcome = None
    exit_price = entry_price
    exit_time = entry_time

    post_entry = daydf[daydf['Date'] > entry_time]

    for idx2, row2 in post_entry.iterrows():
        high2 = row2['High']
        low2 = row2['Low']
        time2 = row2['Date']
        if direction == 'Long':
            if low2 <= sl_price:
                trade_outcome = 'Loss'
                exit_price = sl_price
                exit_time = time2
                debug_print(args, f"Stopped out (loss) at {exit_price} on {exit_time}")
                break
            if high2 >= pt_price:
                trade_outcome = 'Win'
                exit_price = pt_price
                exit_time = time2
                debug_print(args, f"Hit profit target (win) at {exit_price} on {exit_time}")
                break
        else:
            if high2 >= sl_price:
                trade_outcome = 'Loss'
                exit_price = sl_price
                exit_time = time2
                debug_print(args, f"Stopped out (loss) at {exit_price} on {exit_time}")
                break
            if low2 <= pt_price:
                trade_outcome = 'Win'
                exit_price = pt_price
                exit_time = time2
                debug_print(args, f"Hit profit target (win) at {exit_price} on {exit_time}")
                break
    
    if trade_outcome is None:
        if post_entry.empty:
            final_close = entry_price
            final_time = entry_time
        else:
            final_close = post_entry.iloc[-1]['Close']
            final_time = post_entry.iloc[-1]['Date']
        exit_price = final_close
        exit_time = final_time
        if direction == 'Long':
            trade_outcome = 'Win' if exit_price > entry_price else 'Loss'
        else:
            trade_outcome = 'Win' if exit_price < entry_price else 'Loss'
        debug_print(args, f"No exit hit, using close price {exit_price} at {exit_time} as exit, outcome: {trade_outcome}")

    pnl = (exit_price - entry_price) if direction == 'Long' else (entry_price - exit_price)

    debug_print(args, f"Trade PnL: {pnl:.4f}")
    
    result = {
        'Date': entry_time.date(),
        'Direction': direction,
        'EntryPrice': entry_price,
        'ExitPrice': exit_price,
        'Outcome': trade_outcome,
        'EntryTime': entry_time.strftime('%Y-%m-%d %H:%M'),
        'ExitTime': exit_time.strftime('%Y-%m-%d %H:%M'),
        'PnL': round(pnl, 4)
    }
    return result

def backtest_file(filename, year, args):
    df = load_data(filename, year)
    if df.empty:
        if args.debug:
            print(f"[DEBUG] No data or data filtered out for {filename}, year={year}")
        return [], None
    trades = []
    for date, daydf in df.groupby(df['Date'].dt.date):
        result = process_day(daydf, args)
        if result:
            trades.append(result)
    if not trades:
        return trades, None
    total_trades = len(trades)
    wins = sum(t['Outcome'] == 'Win' for t in trades)
    losses = sum(t['Outcome'] == 'Loss' for t in trades)
    avg_pnl = np.mean([t['PnL'] for t in trades])
    total_pnl = np.sum([t['PnL'] for t in trades])
    
    summary = {
        'File': os.path.basename(filename),
        'Year': int(year) if year else 'All',
        'TotalTrades': total_trades,
        'Wins': wins,
        'Losses': losses,
        'WinPct': round(100 * wins / total_trades, 2),
        'AvgPnL': round(avg_pnl, 4),
        'TotalPnL': round(total_pnl, 4)
    }
    return trades, summary

def main():
    args = parse_cli()
    files = []
    if args.file:
        files = [args.file]
        years = [args.year] if args.year else [None]
    elif args.dir:
        files = sorted(glob(os.path.join(args.dir, '*.csv')))
        years = None  # will determine inside
    else:
        print('Please provide --file filename or --dir folder')
        return

    final_summaries = []

    if files and years is not None:
        # Single file mode (with optional year filter)
        for year in years:
            trades, summary = backtest_file(files[0], year, args)
            if trades:
                print(f'\nTrades for {os.path.basename(files[0])}' + (f' Year: {year}' if year else ''))
                print(tabulate(trades, headers='keys', tablefmt='psql', floatfmt=".4f"))
            if summary:
                print('\nSummary:')
                print(tabulate([summary], headers='keys', tablefmt='psql', floatfmt=".4f"))
    else:
        # Directory mode: multiple files, multiple years per file
        for f in files:
            df = pd.read_csv(f)
            df['Date'] = pd.to_datetime(df['Date'])
            file_years = [args.year] if args.year else sorted(df['Date'].dt.year.unique())
            for year in file_years:
                trades, summary = backtest_file(f, year, args)
                if summary:
                    # Progress summary per stock-year
                    print('\nProgress Summary for:', os.path.basename(f), "Year:", year)
                    print(tabulate([summary], headers='keys', tablefmt='psql', floatfmt=".4f"))
                    final_summaries.append(summary)

        # Final combined summary sorted by WinPct descending
        if final_summaries:
            print('\n\n=== FINAL COMBINED SUMMARY FOR ALL STOCKS/YEARS ===')
            df_summary = pd.DataFrame(final_summaries)
            df_summary['Year'] = df_summary['Year'].astype(int)
            df_summary_sorted = df_summary.sort_values(by='WinPct', ascending=False)
            print(tabulate(df_summary_sorted, headers='keys', tablefmt='psql', floatfmt=".4f"))

if __name__ == '__main__':
    main()

