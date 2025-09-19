import pandas as pd
import numpy as np
import argparse

def load_and_prepare_data(filename):
    """
    Automatically parse and clean the input CSV file to return a DataFrame with columns:
    Date (datetime), Open, High, Low, Close (floats).
    
    Supports:
    - Old format (standard OHLC columns with normal numeric)
    - New format (like TCSHistoricalDataSample.csv) with Close first, etc., prices as strings with commas
    """
    df = pd.read_csv(filename)
    cols_lower = [c.strip().lower() for c in df.columns]

    # Detect "new" format: presence of 'vol' column and all required price cols
    new_format_cols = {'date', 'close', 'open', 'high', 'low'}
    new_format_detect = (set(cols_lower) >= new_format_cols) and ('vol' in cols_lower)

    if new_format_detect:
        # Map columns to proper normalized names to avoid case/order issues
        col_map = {}
        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in ('date', 'open', 'high', 'low', 'close'):
                col_map[col] = col_lower.capitalize()
        df = df.rename(columns=col_map)

        # Keep only necessary columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close']]

        # Parse date in DD-MM-YYYY format
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

        def clean_num(val):
            if pd.isna(val):
                return np.nan
            try:
                # Convert string with commas and possible quotes to float
                return float(str(val).replace(',', '').replace('"', '').strip())
            except Exception:
                return np.nan

        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col].apply(clean_num)

    elif set(cols_lower) >= {'date', 'open', 'high', 'low', 'close'}:
        # Old or standard format - try auto date parsing
        df.columns = [c.strip() for c in df.columns]  # strip spaces

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Ensure numeric prices are floats (clean if strings)
        def clean_num_old(val):
            if pd.isna(val):
                return np.nan
            try:
                return float(str(val).replace(',', '').replace('"', '').strip())
            except Exception:
                return np.nan

        for col in ['Open', 'High', 'Low', 'Close']:
            if df[col].dtype == object:
                df[col] = df[col].apply(clean_num_old)

    else:
        raise ValueError("Input CSV format not recognized. Required columns: Date, Open, High, Low, Close.")

    df = df.sort_values('Date').reset_index(drop=True)
    return df

def backtest_3ll_baby_flexible_entry(
        trailing_amount=30,
        filter_year=None,
        summary=False,
        debug=False,
        filename='nifty50sample.txt'
    ):
    df = load_and_prepare_data(filename)
    trades = []
    summary_rows = []

    pattern_count = 0
    baby_candle_count = 0
    both_patterns_count = 0
    trade_triggered_count = 0
    max_entry_wait_days = 3  # Entry allowed up to 3 days after pattern

    for idx in range(3, len(df) - max_entry_wait_days):
        lows = df['Low'].iloc[idx-3:idx+1].reset_index(drop=True)
        highs = df['High'].iloc[idx-3:idx+1].reset_index(drop=True)
        dates = df['Date'].iloc[idx-3:idx+1].reset_index(drop=True)

        baby_candle_date = dates.iloc[3]
        baby_candle_year = baby_candle_date.year

        if filter_year is not None and baby_candle_year != filter_year:
            continue

        cond_3ll = (lows.iloc[1] < lows.iloc[0]) and (lows.iloc[2] < lows.iloc[1])
        cond_baby = (highs.iloc[3] < highs.iloc[2]) and (lows.iloc[3] > lows.iloc[2])
        both_matched = cond_3ll and cond_baby

        if cond_3ll:
            pattern_count += 1
        if cond_baby:
            baby_candle_count += 1
        if both_matched:
            both_patterns_count += 1

        DD_long = np.sqrt(highs.iloc[3]) * 0.2611 if both_matched else np.nan
        entry_price = highs.iloc[3] + DD_long if both_matched else np.nan
        stop_loss = lows.iloc[3] - DD_long if both_matched else np.nan

        entry_triggered = False
        exit_price = np.nan
        profit = np.nan
        exit_reason = ''

        # === CRITICAL FIX: Only attempt entries if BOTH patterns matched ===
        if both_matched:
            entry_day_idx = None
            entry_trigger_found = False
            # Check entry for next 3 days without breaking stop loss before entry
            for offset in range(1, max_entry_wait_days + 1):
                current_day_idx = idx + offset
                if current_day_idx >= len(df):
                    break

                day_high = df.at[current_day_idx, 'High']
                day_low = df.at[current_day_idx, 'Low']
                day_date = df.at[current_day_idx, 'Date']

                if debug and (filter_year is None or baby_candle_year == filter_year):
                    print(f"Checking entry day {day_date.date()} (offset {offset} after baby candle): HL= {day_high:.2f}/{day_low:.2f}")

                # If price hits stop loss before entry, entry opportunity lost
                if day_low <= stop_loss:
                    if debug and (filter_year is None or baby_candle_year == filter_year):
                        print(f"  Stop loss {stop_loss:.2f} hit on {day_date.date()} before entry. Entry lost.")
                    break

                # Entry triggered if high crosses entry price
                if day_high >= entry_price:
                    entry_day_idx = current_day_idx
                    entry_trigger_found = True
                    if debug and (filter_year is None or baby_candle_year == filter_year):
                        print(f"  Entry triggered on {day_date.date()} at entry price {entry_price:.2f}")
                    break

            if entry_trigger_found:
                entry_triggered = True
                trade_triggered_count += 1

                # Initialize multi-increment profit booking logic
                booked_profit_level = entry_price
                long_target = entry_price + 50
                current_stop_loss = stop_loss
                max_favorable_price = max(entry_price, df.at[entry_day_idx, 'High'])

                trade_exit = False
                exit_idx = entry_day_idx
                entry_date = df.at[entry_day_idx, 'Date']

                # Record pattern initiation row
                summary_rows.append({
                    'Date': baby_candle_date.date(),
                    'Event_Type': 'Pattern Initiation',
                    '3LL_Matched': cond_3ll,
                    'Baby_Candle_Matched': cond_baby,
                    'Both_Patterns_Matched': both_matched,
                    'Entry_Triggered': entry_triggered,
                    'DD_Long': round(DD_long, 2),
                    'Entry_Price': round(entry_price, 2),
                    'Exit_Price': '',
                    'Profit': '',
                    'Exit_Reason': '',
                    'Profit_Booked_Level': '',
                    'Trailing_Stop': ''
                })

                if debug:
                    print(f"\nTrade ENTRY at {entry_price:.2f} on {entry_date.date()}")
                    print(f"Stop Loss: {current_stop_loss:.2f}, Trailing Amount: {trailing_amount}")

                # Ride the trade day-by-day after entry until exit
                for day in range(entry_day_idx, len(df)):
                    day_high = df.at[day, 'High']
                    day_low = df.at[day, 'Low']
                    day_close = df.at[day, 'Close']
                    day_date = df.at[day, 'Date']

                    # Update max favorable price
                    if day_high > max_favorable_price:
                        max_favorable_price = day_high

                    # Count how many 50 point increments crossed since last booking
                    increments_crossed = 0
                    while max_favorable_price >= long_target:
                        booked_profit_level = long_target
                        increments_crossed += 1
                        long_target += 50

                    # Log each profit booking event separately, even multiple in same day
                    for i in range(increments_crossed):
                        profit_level_for_row = booked_profit_level - 50 * (increments_crossed - 1 - i)
                        trailing_stop_value = max(max_favorable_price - trailing_amount, booked_profit_level)
                        trailing_stop_value = max(trailing_stop_value, entry_price)

                        summary_rows.append({
                            'Date': day_date.date(),
                            'Event_Type': 'Profit Booking',
                            '3LL_Matched': '',
                            'Baby_Candle_Matched': '',
                            'Both_Patterns_Matched': '',
                            'Entry_Triggered': '',
                            'DD_Long': '',
                            'Entry_Price': '',
                            'Exit_Price': '',
                            'Profit': '',
                            'Exit_Reason': '',
                            'Profit_Booked_Level': round(profit_level_for_row, 2),
                            'Trailing_Stop': round(trailing_stop_value, 2)
                        })

                        if debug:
                            print(f"  Profit booked increment #{i+1} at {profit_level_for_row:.2f} on {day_date.date()}")
                            print(f"  Trailing stop now at {trailing_stop_value:.2f}")

                    trailing_stop = max(max_favorable_price - trailing_amount, booked_profit_level)
                    trailing_stop = max(trailing_stop, entry_price)

                    if debug:
                        print(f"  {day_date.date()} H:{day_high:.2f} L:{day_low:.2f} Close:{day_close:.2f} | MaxFav:{max_favorable_price:.2f} TrailingStop:{trailing_stop:.2f} BookedProfitLvl:{booked_profit_level:.2f}")

                    # Exit conditions priority

                    if day_low <= current_stop_loss:
                        exit_price = current_stop_loss
                        exit_reason = 'stop loss'
                        exit_idx = day
                        trade_exit = True
                        if debug:
                            print(f"    Stop loss hit at {exit_price:.2f} on {day_date.date()}")
                        break

                    elif day_low <= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = 'trailing stop'
                        exit_idx = day
                        trade_exit = True
                        if debug:
                            print(f"    Trailing stop hit at {exit_price:.2f} on {day_date.date()}")
                        break

                    if day == len(df) - 1:
                        exit_price = day_close
                        exit_reason = 'close exit'
                        exit_idx = day
                        trade_exit = True
                        if debug:
                            print(f"    Exit at close {exit_price:.2f} on {day_date.date()} (last data day)")

                # Calculate profit
                profit = exit_price - entry_price
                exit_date = df.at[exit_idx, 'Date']

                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': exit_date,
                    'Trade Type': 'LONG',
                    'Entry Price': round(entry_price, 2),
                    'Exit Price': round(exit_price, 2),
                    'Exit Reason': exit_reason,
                    'Profit': round(profit, 2),
                    'Stop Loss': round(stop_loss, 2),
                    'Trailing Stop': round(trailing_amount, 2)
                })

            else:
                # No entry triggered in 3-day window or SL hit before entry
                summary_rows.append({
                    'Date': baby_candle_date.date(),
                    'Event_Type': 'Pattern Initiation (No Entry)',
                    '3LL_Matched': cond_3ll,
                    'Baby_Candle_Matched': cond_baby,
                    'Both_Patterns_Matched': both_matched,
                    'Entry_Triggered': False,
                    'DD_Long': round(DD_long, 2),
                    'Entry_Price': round(entry_price, 2),
                    'Exit_Price': '',
                    'Profit': '',
                    'Exit_Reason': '',
                    'Profit_Booked_Level': '',
                    'Trailing_Stop': ''
                })
        else:
            # Pattern not matched or partial, still record pattern initiation row
            summary_rows.append({
                'Date': baby_candle_date.date(),
                'Event_Type': 'Pattern Initiation (No Entry)',
                '3LL_Matched': cond_3ll,
                'Baby_Candle_Matched': cond_baby,
                'Both_Patterns_Matched': both_matched,
                'Entry_Triggered': '',
                'DD_Long': round(DD_long, 2) if not np.isnan(DD_long) else '',
                'Entry_Price': round(entry_price, 2) if not np.isnan(entry_price) else '',
                'Exit_Price': '',
                'Profit': '',
                'Exit_Reason': '',
                'Profit_Booked_Level': '',
                'Trailing_Stop': ''
            })

        if debug and (filter_year is None or baby_candle_year == filter_year):
            print(f"\nWindow ending {baby_candle_date.date()} (idx {idx}):")
            print(f"  3LL matched: {cond_3ll}")
            print(f"  Baby candle matched: {cond_baby}")
            print(f"  Both patterns matched: {both_matched}")
            print(f"  Entry triggered: {entry_triggered}")

    # Create summary and trades DataFrames
    summary_df = pd.DataFrame(summary_rows)
    trades_df = pd.DataFrame(trades)

    # Filter trades if year specified
    if filter_year is not None and not trades_df.empty:
        trades_df = trades_df[trades_df['Entry Date'].dt.year == filter_year]

    # Output results
    if summary or filter_year is not None:
        summary_df['Date'] = pd.to_datetime(summary_df['Date'])
        summary_df = summary_df.sort_values(by=['Date', 'Event_Type'], ascending=[True, True])
        summary_df['Date'] = summary_df['Date'].dt.date
        print("\n--- Detailed Per-Day Profit Booking Summary ---\n")
        print(summary_df.to_string(index=False))
        filename = f'3llbaby_longonly_flexibleentry_summary_'
        filename += str(filter_year) if filter_year else 'all_years'
        filename += '.csv'
        summary_df.to_csv(filename, index=False)
        print(f"\nSummary saved as CSV: {filename}")

    if not summary:
        if trades_df.empty:
            print(f"No trades executed for year {filter_year}." if filter_year else "No trades executed.")
        else:
            print("\n--- Detailed Trades ---\n")
            print(trades_df.sort_values(by='Entry Date').to_string(index=False))
            print(f"\nYear: {filter_year if filter_year else 'All years'}")
            print(f"Total trades: {len(trades_df)}")
            print(f"Total profit: {trades_df['Profit'].sum():.2f}")
            print(f"Average profit/trade: {trades_df['Profit'].mean():.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3LL + Baby Candle backtest with 3-day entry window and multi-increment trailing profit booking with automatic CSV loading.")
    parser.add_argument('--year', type=int, help='Filter summary and trades by year (YYYY)', default=None)
    parser.add_argument('--trail', type=float, help='Trailing stop amount (default 30 points)', default=30)
    parser.add_argument('--summary', action='store_true', help='Show detailed per-date summary including profit booking events')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--csv', type=str, help='CSV file to use for historical data', default='nifty50sample.txt')

    args = parser.parse_args()

    backtest_3ll_baby_flexible_entry(
        trailing_amount=args.trail,
        filter_year=args.year,
        summary=args.summary,
        debug=args.debug,
        filename=args.csv
    )

