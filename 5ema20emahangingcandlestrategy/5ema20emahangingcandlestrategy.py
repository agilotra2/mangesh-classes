import argparse
import pandas as pd
import glob
import os
from datetime import datetime

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

class Trade:
    def __init__(self, direction, entry_date, entry_price, exit_date, exit_price, pnl, hit_target, hit_stoploss, debug_log):
        self.direction = direction
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.pnl = pnl
        self.hit_target = hit_target
        self.hit_stoploss = hit_stoploss
        self.debug_log = debug_log

def parse_args():
    parser = argparse.ArgumentParser(description='5/20EMA Swing Strategy Backtester')
    parser.add_argument('--file', type=str, help='CSV file to analyze')
    parser.add_argument('--year', type=int, help='Year to analyze')
    parser.add_argument('--day', type=str, help='Day to debug (YYYY-MM-DD)')
    parser.add_argument('--dir', type=str, help='Directory of CSV files')
    parser.add_argument('--debug', action='store_true', help='Debug mode for stepwise trade reasoning')
    return parser.parse_args()

def max_drawdown(pnl_series):
    # pnl_series is a pandas Series of cumulative PnL
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdowns = running_max - cumulative
    max_dd = drawdowns.max()  # max absolute drawdown (positive value)
    return max_dd

def _find_entry(df, idx, debug_mode=False):
    logs = []
    entries = []
    candle = df.iloc[idx]
    ema5 = candle['5EMA']
    ema20 = candle['20EMA']
    o, h, l, c = candle['Open'], candle['High'], candle['Low'], candle['Close']
    date = candle['Date'][:10]
    if debug_mode:
        logs.append(f"DAY: {date} | Open:{o} High:{h} Low:{l} Close:{c} | 5EMA:{ema5:.2f} 20EMA:{ema20:.2f}")
    # Entry checks
    is_long = (l > ema5 and h > ema5 and ema5 > ema20)
    is_short = (h < ema5 and l < ema5 and ema5 < ema20)
    if debug_mode:
        logs.append(f"[Check] Candle fully above 5EMA: {'YES' if (l > ema5 and h > ema5) else 'NO'}")
        logs.append(f"[Check] Candle fully below 5EMA: {'YES' if (h < ema5 and l < ema5) else 'NO'}")
        logs.append(f"[Check] 5EMA > 20EMA: {'YES' if ema5 > ema20 else 'NO'}, 5EMA < 20EMA: {'YES' if ema5 < ema20 else 'NO'}")
    # Entry trigger: Next candle breakout
    if is_long or is_short:
        if idx+1 < len(df):
            next_candle = df.iloc[idx+1]
            next_date = next_candle['Date'][:10]
            nh, nl, no = next_candle['High'], next_candle['Low'], next_candle['Open']
            if is_long and nh > h:
                entries.append(('long', idx, next_candle['Date'], no, logs))
                if debug_mode:
                    logs.append(f"Long Entry Triggered: Next candle's high ({nh:.2f}) breached current high ({h:.2f}), entry at {next_date} OPEN={no:.2f}")
            elif is_short and nl < l:
                entries.append(('short', idx, next_candle['Date'], no, logs))
                if debug_mode:
                    logs.append(f"Short Entry Triggered: Next candle's low ({nl:.2f}) breached current low ({l:.2f}), entry at {next_date} OPEN={no:.2f}")
            else:
                if debug_mode:
                    logs.append(f"No breakout for {'LONG' if is_long else 'SHORT'} entry (next high={nh:.2f}, low={nl:.2f})")
    else:
        if debug_mode:
            logs.append("No qualifying entry for this candle.")
    return entries, logs

def backtest(df, args):
    trades = []
    debug_day = args.day
    debug_mode = args.debug
    # Year filter for partial data set if given
    if args.year:
        df = df[df['Date'].str[:4] == str(args.year)]
    df['5EMA'] = calculate_ema(df['Close'], 5)
    df['20EMA'] = calculate_ema(df['Close'], 20)
    last_trade_day = None
    for idx in range(len(df)-1):
        day = df.iloc[idx]['Date'][:10]
        if debug_day and day != debug_day:
            continue
        entries, logs = _find_entry(df, idx, debug_mode)
        for direction, entry_idx, entry_date, entry_price, entry_logs in entries:
            entry_day = entry_date[:10]
            if last_trade_day == entry_day:
                if debug_mode:
                    entry_logs.append(f"Trade for {entry_day} skipped (already took trade that day).")
                continue
            target = entry_price * (1.05 if direction == 'long' else 0.95)
            stop = entry_price * (0.95 if direction == 'long' else 1.05)
            if debug_mode:
                entry_logs.append(f"Entry: {direction.upper()} | Entry date: {entry_day} | Entry price: {entry_price:.2f}")
                entry_logs.append(f"Target: {target:.2f} | Stop: {stop:.2f}")
            outcome = None
            for j in range(entry_idx+1, len(df)):
                row = df.iloc[j]
                ex_date = row['Date'][:10]
                high, low = row['High'], row['Low']
                if debug_mode:
                    entry_logs.append(f"Checking exit bar: {ex_date} | High: {high:.2f}, Low: {low:.2f}")
                if direction == 'long':
                    if low <= stop:
                        outcome = Trade(direction, entry_day, entry_price, row['Date'], stop, stop-entry_price, False, True, entry_logs + [f"Stop loss hit at {ex_date} (Low={low:.2f} <= SL={stop:.2f})"])
                        break
                    if high >= target:
                        outcome = Trade(direction, entry_day, entry_price, row['Date'], target, target-entry_price, True, False, entry_logs + [f"Target hit at {ex_date} (High={high:.2f} >= TP={target:.2f})"])
                        break
                else:
                    if high >= stop:
                        outcome = Trade(direction, entry_day, entry_price, row['Date'], stop, entry_price-stop, False, True, entry_logs + [f"Stop loss hit at {ex_date} (High={high:.2f} >= SL={stop:.2f})"])
                        break
                    if low <= target:
                        outcome = Trade(direction, entry_day, entry_price, row['Date'], target, entry_price-target, True, False, entry_logs + [f"Target hit at {ex_date} (Low={low:.2f} <= TP={target:.2f})"])
                        break
            if outcome:
                trades.append(outcome)
                last_trade_day = entry_day
            else:
                if debug_mode:
                    entry_logs.append("NO EXIT: Neither target nor SL hit within data.")
        if debug_mode and (not entries):
            for logline in logs:
                print(logline)
    return trades

def summarize_trades(trades):
    # return dict with summary fields incl max drawdown and PnL per year
    df = pd.DataFrame([{
        'entry_date': t.entry_date[:10],
        'pnl': t.pnl
    } for t in trades])

    if df.empty:
        return {'total_trades':0, 'wins':0, 'losses':0, 'total_pnl':0.0, 'winrate':0.0, 'max_drawdown':0.0, 'yearly':{}}

    df['year'] = pd.to_datetime(df['entry_date']).dt.year
    total_trades = len(trades)
    wins = len([t for t in trades if t.hit_target])
    losses = len([t for t in trades if t.hit_stoploss])
    total_pnl = df['pnl'].sum()
    winrate = (wins / total_trades) * 100 if total_trades else 0.0

    # Calculate max drawdown on cumulative pnl
    df['cumpnl'] = df['pnl'].cumsum()
    running_max = df['cumpnl'].cummax()
    drawdowns = running_max - df['cumpnl']
    max_dd = drawdowns.max() if not drawdowns.empty else 0.0

    yearly_summary = {}
    for y, group in df.groupby('year'):
        y_total_trades = len(group)
        y_wins = len(group[group['pnl'] > 0])
        y_losses = len(group[group['pnl'] <= 0])
        y_total_pnl = group['pnl'].sum()
        y_winrate = (y_wins / y_total_trades) * 100 if y_total_trades else 0.0
        y_running_max = group['cumpnl'].cummax()
        y_drawdowns = y_running_max - group['cumpnl']
        y_max_dd = y_drawdowns.max() if not y_drawdowns.empty else 0.0

        yearly_summary[y] = {
            'trades': y_total_trades,
            'wins': y_wins,
            'losses': y_losses,
            'pnl': y_total_pnl,
            'winrate': y_winrate,
            'max_drawdown': y_max_dd
        }

    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'total_pnl': total_pnl,
        'winrate': winrate,
        'max_drawdown': max_dd,
        'yearly': yearly_summary
    }

def output_summary(trades, stock_name, year=None, fname=None, debug_mode=False):
    summary = summarize_trades(trades)

    print(f"Summary for {stock_name} {year if year else ''}")
    print(f"Total trades: {summary['total_trades']} | Wins: {summary['wins']} Losses: {summary['losses']} | "
          f"Total PnL: {summary['total_pnl']:.2f} | Winrate: {summary['winrate']:.2f}% | Max Drawdown: {summary['max_drawdown']:.2f}")

    # Yearwise summary if year not specified (multi-year file)
    if not year and summary['yearly']:
        print("\nYear-wise Performance:")
        for y, ysum in sorted(summary['yearly'].items()):
            print(f" {y}: Trades={ysum['trades']} Wins={ysum['wins']} Losses={ysum['losses']} PnL={ysum['pnl']:.2f} "
                  f"Winrate={ysum['winrate']:.2f}% MaxDrawdown={ysum['max_drawdown']:.2f}")

    # CSV detailed output
    if fname:
        df_out = pd.DataFrame([{
            "Date": t.entry_date,
            "Direction": t.direction,
            "Entry": t.entry_price,
            "ExitDate": t.exit_date,
            "Exit": t.exit_price,
            "Pnl": t.pnl,
            "TargetHit": t.hit_target,
            "SLHit": t.hit_stoploss
        } for t in trades])
        df_out.to_csv(fname, index=False)

    if debug_mode:
        print("\n------ DEBUG LOGS PER TRADE ------")
        for t in trades:
            print(f"[Trade {t.entry_date} {t.direction.upper()}]")
            for log in t.debug_log:
                print(log)
            print("-" * 40)

def process_file(file, args, suppress_daywise_print=False):
    stock_name = os.path.basename(file).split('.')[0]
    df = pd.read_csv(file)

    # Normalize columns to lowercase and strip spaces
    df.columns = df.columns.str.strip().str.lower()

    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"WARNING: File '{file}' skipped due to missing columns: {missing_cols}")
        return stock_name, 0, 0, 0, 0.0, 0.0
    
    df = df.rename(columns={
        'date': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    trades = backtest(df, args)
    dtstr = datetime.now().strftime('%Y%m%d_%H%M')
    fname = f"{stock_name}_summary_{dtstr}.csv" if not args.dir else None
    if not suppress_daywise_print:
        output_summary(trades, stock_name, args.year, fname, args.debug)
    winners = sum(1 for t in trades if t.hit_target)
    losers = sum(1 for t in trades if t.hit_stoploss)
    total_pnl = sum(t.pnl for t in trades)
    winrate = (winners / len(trades) * 100) if trades else 0
    return stock_name, len(trades), winners, losers, total_pnl, winrate

def process_dir(folder, args):
    all_summaries = []
    files = glob.glob(os.path.join(folder, '*.csv'))
    for file in files:
        stock, trades_count, winners, losers, total_pnl, winrate = process_file(file, args, suppress_daywise_print=True)
        print(f"{stock}: Trades={trades_count} Wins={winners} Losses={losers} PnL={total_pnl:.2f} WinRate={winrate:.2f}%")
        all_summaries.append({
            'Stock': stock,
            'Year': str(args.year) if args.year else 'ALL',
            'Trades': trades_count,
            'Wins': winners,
            'Losses': losers,
            'TotalPnL': total_pnl,
            'WinRate': winrate
        })
    all_summaries.sort(key=lambda x: x['TotalPnL'], reverse=True)
    dtstr = datetime.now().strftime('%Y%m%d_%H%M')
    outcsv = f"combined_summary_{dtstr}.csv"
    pd.DataFrame(all_summaries).to_csv(outcsv, index=False)
    print("Top 3 stocks by PnL:")
    for row in all_summaries[:3]:
        print(row)

def main():
    args = parse_args()
    if args.dir:
        process_dir(args.dir, args)
    elif args.file:
        process_file(args.file, args)
    else:
        print("Please supply either --file <file> or --dir <folder>")

if __name__ == '__main__':
    main()

