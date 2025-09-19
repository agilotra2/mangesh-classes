import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Find O=H and O=L days in Nifty index data CSV')
parser.add_argument('csvfile', type=str, help='Path to CSV file containing Nifty OHLC data')
args = parser.parse_args()

df = pd.read_csv(args.csvfile)

required_columns = {'Date', 'Open', 'High', 'Low'}
if not required_columns.issubset(set(df.columns)):
    raise ValueError(f"CSV must contain columns: {required_columns}")

oh_days = df[df['Open'] == df['High']]
ol_days = df[df['Open'] == df['Low']]

print("Days where Open = High:")
print(oh_days['Date'].to_list())
print("\nDays where Open = Low:")
print(ol_days['Date'].to_list())

# Write to CSV
oh_days.to_csv("Open_Equals_High.csv", columns=['Date', 'Open', 'High', 'Low'], index=False)
ol_days.to_csv("Open_Equals_Low.csv", columns=['Date', 'Open', 'High', 'Low'], index=False)
print("\nCSV files 'Open_Equals_High.csv' and 'Open_Equals_Low.csv' saved.")

