# Mangesh Classes - Trading Strategies Collection

A comprehensive collection of algorithmic trading strategies developed for the Indian stock market using 5-minute timeframe data.

## 📁 Repository Structure

### 🔥 **Active Strategies**

#### EMA-Based Strategies
- **`5ema20emahangingcandlestrategy/`** - 5 & 20 EMA with hanging candle pattern detection
- **`10ema20emahangingcandlestrategy/`** - 10 & 20 EMA with hanging candle patterns

#### Time-Based Strategies  
- **`index1306strategy/`** - Index trading strategy with 13:06 timing component
- **`11301230period.py`** - Period-based strategy (11:30-12:30)

#### Pivot & Swing Strategies
- **`195minbacktest5mTFstock*.py`** - Multiple versions of 195-minute pivot strategies
- **`amarakbaranthony/`** - Custom swing strategy

#### Pattern Recognition
- **`3llbabybacktest*.py`** - "3LL Baby" pattern recognition strategies
- **`closeeqopenandonehourhigh.py`** - Open=Close and one-hour high patterns
- **`intradaybias.py`** - Intraday bias detection
- **`ohol.py`** - Open high, open low patterns

### 📊 **Data Sources**
- **`JioAICloud-AllParts-5mTFAllStocks/`** - Complete 5-minute data for all stocks (1,431+ files)
- **`JioAICloud-Download-5mTFStocks/`** - Downloaded stock data (427+ files)  
- **Sample data files**: NIFTY-50, TCS, Raymond historical data

### 📈 **Results & Performance**
- **Backtest results**: CSV files with strategy performance metrics
- **Trade logs**: Detailed trade-by-trade analysis
- **Performance summaries**: Yearly and overall strategy statistics

## 🚀 **Key Features**

- **Multi-timeframe analysis**: Primarily 5-minute data with higher timeframe filters
- **Pattern recognition**: EMA crossovers, hanging candles, pivot points
- **Risk management**: Stop-loss and target implementations
- **Backtesting framework**: Comprehensive historical testing
- **Performance tracking**: Detailed metrics and reporting

## 📋 **Strategy Categories**

1. **Momentum Strategies**: EMA-based trend following
2. **Mean Reversion**: Pivot and swing-based approaches  
3. **Pattern Recognition**: Candlestick and price action patterns
4. **Time-based**: Specific trading hour strategies
5. **Multi-factor**: Combined technical indicators

## 🔧 **Usage**

Most strategies follow this pattern:
```bash
python strategy_name.py --data_path /path/to/5min/data --year 2024
```

## 📊 **Data Requirements**

- **Format**: CSV files with OHLCV data
- **Timeframe**: 5-minute bars
- **Timezone**: Asia/Kolkata
- **Columns**: Date, Open, High, Low, Close, Volume

## ⚠️ **Important Notes**

- Strategies are designed for Indian stock market (NSE/BSE)
- All times are in IST (Indian Standard Time)
- Backtesting includes transaction costs and realistic slippage
- Results are for educational and research purposes

## 🔄 **Recent Updates**

- Active development through September 2024
- Multiple strategy variations and optimizations
- Extensive backtesting across different market conditions
- Performance analysis across multiple years (2015-2025)

---

*This repository contains proprietary trading strategies developed for educational purposes. Past performance does not guarantee future results.*