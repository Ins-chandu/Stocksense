import pandas as pd
import time
from alpha_vantage.timeseries import TimeSeries
import ta
import yfinance as yf
from app.config import API_KEY


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_diff"] = macd.macd_diff()
    df["SMA20"] = ta.trend.SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["SMA200"] = ta.trend.SMAIndicator(close=df["Close"], window=200).sma_indicator()
    df["RET1"] = df["Close"].pct_change(1)
    df["VOL"] = df["RET1"].rolling(window=10).std()
    df.dropna(inplace=True)
    return df


def _fetch_from_alpha_vantage(symbol: str) -> pd.DataFrame:
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    data, _ = ts.get_daily(symbol=symbol, outputsize="full")
    time.sleep(12)

    df = data[["1. open", "2. high", "3. low", "4. close", "5. volume"]].copy()
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df[::-1]


def _fetch_from_yfinance(symbol: str) -> pd.DataFrame:
    ticker = symbol.replace(".BSE", ".BO")
    data = yf.download(
        ticker, period="10y", interval="1d", auto_adjust=False, progress=False
    )
    if data.empty:
        raise ValueError(f"No market data found for symbol {symbol}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data[["Open", "High", "Low", "Close", "Volume"]].copy()


def fetch_data(symbol: str) -> pd.DataFrame:
    try:
        df = _fetch_from_alpha_vantage(symbol)
    except Exception:
        df = _fetch_from_yfinance(symbol)

    feature_df = _build_feature_frame(df)
    if feature_df.empty:
        raise ValueError(f"Not enough historical data available for {symbol}.")
    return feature_df
