import os
import argparse
import pandas as pd
from datetime import datetime

def process_stock_file(filepath, year_filter=None):
    stock = os.path.splitext(os.path.basename(filepath))[0]
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df['date_only'] = df['Date'].dt.date
    df['year'] = df['Date'].dt.year

    if year_filter:
        df = df[df['year'] == int(year_filter)]

    years_present = df['year'].unique()
    years_str = ", ".join(map(str, sorted(years_present))) if len(years_present) > 0 else "No data"

    results = []
    for date, group in df.groupby('date_only'):
        group = group.sort_values('Date')
        if len(group) >= 9:
            price_45min = group.iloc[8]['Close']
            close_eod = group.iloc[-1]['Close']
            if abs(price_45min) < 1e-10:
                continue
            post_45min_return = (close_eod - price_45min) / price_45min * 100
            results.append(post_45min_return)

    if results:
        avg_return = sum(results) / len(results)
        pct_positive = sum(r > 0 for r in results) / len(results) * 100
        count_days = len(results)
    else:
        avg_return = float('nan')
        pct_positive = float('nan')
        count_days = 0

    if pct_positive > 75:
        suggested_strategy = "Go Long"
    elif pct_positive < 20:
        suggested_strategy = "Go Short"
    else:
        suggested_strategy = "No Clear Bias"

    return {
        'Stock': stock,
        'Years_Processed': years_str,
        'Avg_Post45_Return (%)': round(avg_return, 4),
        'Pct_Positive_Days (%)': round(pct_positive, 2),
        'Days_Analyzed': count_days,
        'Suggested_Strategy': suggested_strategy
    }

def main():
    parser = argparse.ArgumentParser(description='Calculate post-45min intraday tendencies for stocks.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dir', help='Directory containing stock CSV files')
    group.add_argument('--file', help='Single stock CSV file to process')
    parser.add_argument('--year', required=False, help='Year filter (optional, e.g. 2019)')
    parser.add_argument('--output', required=False, default='summary.csv', help='Output summary CSV filename')
    args = parser.parse_args()

    summaries = []

    if args.dir:
        stock_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('.csv')]
        total_files = len(stock_files)
        for idx, filepath in enumerate(stock_files, 1):
            # Read file first to get years info without heavy processing
            temp_df = pd.read_csv(filepath, usecols=['Date'], parse_dates=['Date'])
            years_in_file = temp_df['Date'].dt.year.unique()
            if args.year:
                years_processed = [year for year in years_in_file if year == int(args.year)]
            else:
                years_processed = years_in_file
            years_str = ", ".join(map(str, sorted(years_processed))) if len(years_processed) > 0 else "No data"

            print(f"Processing stock {idx} of {total_files}: {os.path.basename(filepath)}, Years: {years_str}")
            summary = process_stock_file(filepath, args.year)
            summaries.append(summary)
    elif args.file:
        temp_df = pd.read_csv(args.file, usecols=['Date'], parse_dates=['Date'])
        years_in_file = temp_df['Date'].dt.year.unique()
        if args.year:
            years_processed = [year for year in years_in_file if year == int(args.year)]
        else:
            years_processed = years_in_file
        years_str = ", ".join(map(str, sorted(years_processed))) if len(years_processed) > 0 else "No data"

        print(f"Processing single file: {os.path.basename(args.file)}, Years: {years_str}")
        summary = process_stock_file(args.file, args.year)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output, index=False)

    print("\nFinal Summary:")
    print(summary_df.to_string(index=False))
    print(f'\nResults saved to {args.output}')

if __name__ == '__main__':
    main()

