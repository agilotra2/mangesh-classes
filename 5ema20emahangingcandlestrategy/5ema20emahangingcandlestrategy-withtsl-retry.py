import argparse
import pandas as pd
import pandas_ta as ta
import glob
import os
from datetime import datetime

def calculate_ema(df, period, price_col='Close'):
    # Use pandas_ta EMA calculation, returns Series
    return ta.ema(df[price_col], length=period)

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
    parser.add_argument('--file', type=str, help='CSV file to analyze (single stock multi-year)')
    parser.add_argument('--year', type=int, help='Specific year to analyze')
    parser.add_argument('--day', type=str, help='Specific day to debug (YYYY-MM-DD)')
    parser.add_argument('--dir', type=str, help='Directory containing stock CSV files')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logs')
    return parser.parse_args()

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
    is_long = (l > ema5 and h > ema5 and ema5 > ema20)
    is_short = (h < ema5 and l < ema5 and ema5 < ema20)
    if debug_mode:
        logs.append(f"[Check] Candle fully above 5EMA: {'YES' if (l > ema5 and h > ema5) else 'NO'}")
        logs.append(f"[Check] Candle fully below 5EMA: {'YES' if (h < ema5 and l < ema5) else 'NO'}")
        logs.append(f"[Check] 5EMA > 20EMA: {'YES' if ema5 > ema20 else 'NO'}, 5EMA < 20EMA: {'YES' if ema5 < ema20 else 'NO'}")
    if is_long or is_short:
        if idx+1 < len(df):
            next_candle = df.iloc[idx+1]
            next_date = next_candle['Date'][:10]
            nh, nl, no = next_candle['High'], next_candle['Low'], next_candle['Open']
            if is_long and nh > h:
                entry_price = max(h, no)
                entries.append(('long', idx, next_candle['Date'], entry_price, logs))
                if debug_mode:
                    logs.append(f"Long Entry Triggered: breakout price {h:.2f}, next open {no:.2f}, entry assumed at {entry_price:.2f} on {next_date}")
            elif is_short and nl < l:
                entry_price = min(l, no)
                entries.append(('short', idx, next_candle['Date'], entry_price, logs))
                if debug_mode:
                    logs.append(f"Short Entry Triggered: breakout price {l:.2f}, next open {no:.2f}, entry assumed at {entry_price:.2f} on {next_date}")
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
    if args.year:
        df = df[df['Date'].str[:4] == str(args.year)]

    df['5EMA'] = calculate_ema(df, 5)
    df['20EMA'] = calculate_ema(df, 20)

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

            if direction == 'long':
                target_perc = 0.05  
                stop_perc = 0.05
                curr_target = entry_price * (1 + target_perc)
                curr_stop = entry_price * (1 - stop_perc)
            else:
                target_perc = 0.05
                stop_perc = 0.05
                curr_target = entry_price * (1 - target_perc)
                curr_stop = entry_price * (1 + stop_perc)

            accrued_pnl = 0.0

            if debug_mode:
                entry_logs.append(f"Entry: {direction.upper()} | Entry date: {entry_day} | Entry price: {entry_price:.2f}")
                entry_logs.append(f"Initial Target: {curr_target:.2f} (+{target_perc*100:.1f}%) | Initial Stop: {curr_stop:.2f} (-{stop_perc*100:.1f}%)")

            outcome = None
            for j in range(entry_idx+1, len(df)):
                row = df.iloc[j]
                ex_date = row['Date'][:10]
                high, low = row['High'], row['Low']

                if debug_mode:
                    entry_logs.append(f"Checking exit bar: {ex_date} | High: {high:.2f}, Low: {low:.2f} | Target: {curr_target:.2f}, Stop: {curr_stop:.2f} | Accrued PnL: {accrued_pnl:.2f}")

                if direction == 'long':
                    if low <= curr_stop:
                        total_pnl = accrued_pnl + (curr_stop - entry_price)
                        entry_logs.append(f"Stop loss hit at {ex_date} (Low={low:.2f} <= SL={curr_stop:.2f}). Final PnL: {total_pnl:.2f}")
                        outcome = Trade(direction, entry_day, entry_price, row['Date'], curr_stop, total_pnl, total_pnl > 0, total_pnl <= 0, entry_logs)
                        break
                    if high >= curr_target:
                        old_target = curr_target
                        old_stop = curr_stop
                        accrued_pnl += (curr_target - entry_price) - accrued_pnl
                        target_perc += 0.03
                        stop_perc += 0.03
                        curr_target = entry_price * (1 + target_perc)
                        curr_stop = old_stop + (entry_price * 0.03)
                        entry_logs.append(f"Target hit at {ex_date} (High={high:.2f}), ratcheted target to {curr_target:.2f} and stop to {curr_stop:.2f}, accrued PnL: {accrued_pnl:.2f}")
                        continue

                else:
                    if high >= curr_stop:
                        total_pnl = accrued_pnl + (entry_price - curr_stop)
                        entry_logs.append(f"Stop loss hit at {ex_date} (High={high:.2f} >= SL={curr_stop:.2f}). Final PnL: {total_pnl:.2f}")
                        outcome = Trade(direction, entry_day, entry_price, row['Date'], curr_stop, total_pnl, total_pnl > 0, total_pnl <= 0, entry_logs)
                        break
                    if low <= curr_target:
                        old_target = curr_target
                        old_stop = curr_stop
                        accrued_pnl += (entry_price - curr_target) - accrued_pnl
                        target_perc += 0.03
                        stop_perc += 0.03
                        curr_target = entry_price * (1 - target_perc)
                        curr_stop = old_stop - (entry_price * 0.03)
                        entry_logs.append(f"Target hit at {ex_date} (Low={low:.2f}), ratcheted target to {curr_target:.2f} and stop to {curr_stop:.2f}, accrued PnL: {accrued_pnl:.2f}")
                        continue

            if outcome:
                trades.append(outcome)
                last_trade_day = entry_day
            else:
                if debug_mode:
                    entry_logs.append("NO EXIT: Neither target nor stop loss hit within data.")
        if debug_mode and (not entries):
            for log_line in logs:
                print(log_line)
    return trades

def summarize_trades(trades):
    if not trades:
        return {
            'total_trades': 0, 'wins': 0, 'losses': 0,
            'total_pnl': 0.0, 'winrate': 0.0, 'max_drawdown': 0.0,
            'yearly': {}, 'daily': {}
        }
    df = pd.DataFrame([{
        'entry_date': t.entry_date[:10],
        'exit_date': t.exit_date[:10],
        'pnl': t.pnl,
        'hit_target': t.hit_target
    } for t in trades])

    df['year'] = pd.to_datetime(df['exit_date']).dt.year
    df['date'] = pd.to_datetime(df['exit_date']).dt.date

    total_trades = len(trades)
    wins = (df['pnl'] > 0).sum()
    losses = (df['pnl'] <= 0).sum()
    total_pnl = df['pnl'].sum()
    winrate = (wins / total_trades) * 100 if total_trades else 0.0

    df['cumpnl'] = df['pnl'].cumsum()
    running_max = df['cumpnl'].cummax()
    drawdowns = running_max - df['cumpnl']
    max_dd = drawdowns.max() if not drawdowns.empty else 0.0

    yearly_summary = {}
    for y, group in df.groupby('year'):
        y_total_trades = len(group)
        y_wins = (group['pnl'] > 0).sum()
        y_losses = (group['pnl'] <= 0).sum()
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

    daily_summary = {}
    for d, group in df.groupby('date'):
        d_total_trades = len(group)
        d_wins = (group['pnl'] > 0).sum()
        d_losses = (group['pnl'] <= 0).sum()
        d_total_pnl = group['pnl'].sum()
        d_winrate = (d_wins / d_total_trades) * 100 if d_total_trades else 0.0
        daily_summary[str(d)] = {
            'trades': d_total_trades,
            'wins': d_wins,
            'losses': d_losses,
            'pnl': d_total_pnl,
            'winrate': d_winrate
        }

    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'total_pnl': total_pnl,
        'winrate': winrate,
        'max_drawdown': max_dd,
        'yearly': yearly_summary,
        'daily': daily_summary
    }

def output_summary(trades, stock_name, year=None, fname=None, debug_mode=False):
    summary = summarize_trades(trades)

    print(f"Summary for {stock_name} {year if year else ''}")
    print(f"Total trades: {summary['total_trades']} | Wins: {summary['wins']} Losses: {summary['losses']} | "
          f"Total PnL: {summary['total_pnl']:.2f} | Winrate: {summary['winrate']:.2f}% | Max Drawdown: {summary['max_drawdown']:.2f}")

    if not year and summary['yearly']:
        print("\nYear-wise Performance:")
        for y, ys in sorted(summary['yearly'].items()):
            print(f" {y}: Trades={ys['trades']} Wins={ys['wins']} Losses={ys['losses']} "
                  f"PnL={ys['pnl']:.2f} Winrate={ys['winrate']:.2f}% MaxDrawdown={ys['max_drawdown']:.2f}")

    if year and debug_mode:
        print("\nDay-wise Performance (for debug and --year):")
        for d, ds in sorted(summary['daily'].items()):
            print(f"{d}: Trades={ds['trades']} Wins={ds['wins']} Losses={ds['losses']} "
                  f"PnL={ds['pnl']:.2f} Winrate={ds['winrate']:.2f}%")

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

def prepare_df_dataframe(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"File '{file}' missing required columns: {missing_cols}")
    df = df.rename(columns={
        'date': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return df

def process_file(file, args, suppress_daywise_print=False):
    stock_name = os.path.basename(file).split('.')[0]
    try:
        df = prepare_df_dataframe(file)
    except ValueError as e:
        print(f"WARNING: {e}. Skipping file.")
        return stock_name, 0, 0, 0, 0.0, 0.0

    trades = backtest(df, args)
    dtstr = datetime.now().strftime('%Y%m%d_%H%M')
    fname = f"{stock_name}_summary_{dtstr}.csv" if not args.dir else None
    show_daywise = (args.year is not None and args.debug)
    if not suppress_daywise_print or show_daywise:
        output_summary(trades, stock_name, args.year, fname, args.debug)
    winners = sum(1 for t in trades if t.pnl > 0)
    losers = sum(1 for t in trades if t.pnl <= 0)
    total_pnl = sum(t.pnl for t in trades)
    winrate = (winners / len(trades) * 100) if trades else 0
    return stock_name, len(trades), winners, losers, total_pnl, winrate

def process_dir(folder, args):
    all_summaries = []
    files = glob.glob(os.path.join(folder, '*.csv'))
    print(f"Processing directory: {folder}")
    print(f"Found {len(files)} stock files to backtest.\n")

    for idx, file in enumerate(files, start=1):
        stock_name = os.path.basename(file).split('.')[0]
        print(f"Processing file {idx} of {len(files)}: {stock_name}.csv")

        try:
            df = prepare_df_dataframe(file)
        except ValueError as e:
            print(f"WARNING: {e}. Skipping file.")
            continue

        trades = backtest(df, args)
        summary = summarize_trades(trades)
        yearly = summary.get('yearly', {})

        if yearly:
            print(" Year-wise Performance:")
            for year, stats in sorted(yearly.items()):
                print(f"{stock_name} {year}: Trades={stats['trades']} Wins={stats['wins']} Losses={stats['losses']} "
                      f"PnL={stats['pnl']:.2f} WinRate={stats['winrate']:.2f}%")
        else:
            print(f"{stock_name}: No trades found.")

        # Accumulate combined summary across years for leaderboard
        winners = summary['wins']
        losers = summary['losses']
        total_trades = summary['total_trades']
        total_pnl = summary['total_pnl']
        winrate = summary['winrate']

        all_summaries.append({
            'Stock': stock_name,
            'Trades': total_trades,
            'Wins': winners,
            'Losses': losers,
            'TotalPnL': total_pnl,
            'WinRate': winrate
        })

        print("")

    leaderboard = sorted(all_summaries, key=lambda x: x['TotalPnL'], reverse=True)
    dtstr = datetime.now().strftime('%Y%m%d_%H%M')
    outcsv = f"combined_summary_{dtstr}.csv"
    pd.DataFrame(leaderboard).to_csv(outcsv, index=False)

    print("Top 3 stocks by total PnL:")
    for i, entry in enumerate(leaderboard[:3], 1):
        print(f"{i}. {entry['Stock']}: Trades={entry['Trades']} Wins={entry['Wins']} Losses={entry['Losses']} PnL={entry['TotalPnL']:.2f} WinRate={entry['WinRate']:.2f}%")

    print(f"Summary CSV saved as {outcsv}")

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

