import pandas as pd

# Load data
df = pd.read_csv('nifty50sample.txt')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Target dates
target_dates = pd.to_datetime(["2023-05-30", "2023-05-31", "2023-06-01", "2023-06-02"])
#target_dates = pd.to_datetime(["2023-06-21", "2023-06-22", "2023-06-23", "2023-06-26"])

# Filter rows for only these dates
subset = df[df['Date'].isin(target_dates)][['Date', 'Open', 'High', 'Low', 'Close']]

# Check if all dates present
if len(subset) != 4:
    missing_dates = set(target_dates) - set(subset['Date'])
    print("Missing data for dates:", [d.strftime("%Y-%m-%d") for d in missing_dates])
else:
    print("Data for the selected 4 days:")
    print(subset.to_string(index=False))

    lows = subset['Low'].values
    highs = subset['High'].values

    print("\nChecking 3 consecutive lower lows on first 3 candles:")
    cond_3ll = (lows[1] < lows[0]) and (lows[2] < lows[1])
    print(f"Low {subset.iloc[1]['Date'].date()} < Low {subset.iloc[0]['Date'].date()}: {lows[1]:.2f} < {lows[0]:.2f} = {lows[1] < lows[0]}")
    print(f"Low {subset.iloc[2]['Date'].date()} < Low {subset.iloc[1]['Date'].date()}: {lows[2]:.2f} < {lows[1]:.2f} = {lows[2] < lows[1]}")
    print(f"Overall 3 consecutive lower lows condition: {cond_3ll}")

    print("\nChecking baby candle (4th) inside bar condition relative to 3rd candle:")
    baby_high = highs[3]
    baby_low = lows[3]
    prev_high = highs[2]
    prev_low = lows[2]

    cond_baby_inside = (baby_high < prev_high) and (baby_low > prev_low)
    print(f"Baby High {baby_high:.2f} < Prev High {prev_high:.2f}: {baby_high < prev_high}")
    print(f"Baby Low {baby_low:.2f} > Prev Low {prev_low:.2f}: {baby_low > prev_low}")
    print(f"Overall baby candle inside bar condition: {cond_baby_inside}")

    if cond_3ll and cond_baby_inside:
        print("\nPattern detected according to logic.")
    else:
        print("\nPattern NOT detected.")


