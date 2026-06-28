import os
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical stock data from Yahoo Finance and caches it locally as a CSV.
    
    Parameters:
        ticker (str): The stock symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: Stock price data with DatetimeIndex.
    """
    safe_ticker = ticker.replace("^", "INDEX_").replace("=", "EQ_")
    filename = f"{safe_ticker}_{start_date}_{end_date}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    # Check if local cache exists
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
            # In pandas, index name might be capitalized or not depending on yfinance/CSV saving
            if not df.empty:
                return df
        except Exception:
            # If reading cache fails, download fresh data
            pass
            
    # Download fresh data using yfinance
    # Use threads=False to prevent issues in some Streamlit environments
    df = yf.download(ticker, start=start_date, end=end_date, threads=False)
    
    if df.empty:
        raise ValueError(f"No stock data found for ticker '{ticker}' in date range {start_date} to {end_date}.")
    
    # Flatten MultiIndex columns if present (common in yfinance v0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        # Retrieve the main column names (Open, High, Low, Close, etc.)
        df.columns = df.columns.get_level_values(0)
        
    # Ensure standard column naming and index
    if "Close" not in df.columns and "Adj Close" not in df.columns:
        raise ValueError(f"Expected Close or Adj Close in yfinance output, but got: {df.columns.tolist()}")
        
    # Save to local cache
    df.to_csv(filepath)
    return df

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculates the daily logarithmic returns of a stock price series.
    
    Formula: R_t = ln(P_t / P_{t-1})
    
    Parameters:
        prices (pd.Series): Series of closing prices.
        
    Returns:
        pd.Series: Log returns, with the first NaN dropped.
    """
    # Ensure there are no zero/negative prices that would break logarithm
    valid_prices = prices.dropna()
    valid_prices = valid_prices[valid_prices > 0]
    
    log_returns = np.log(valid_prices / valid_prices.shift(1)).dropna()
    return log_returns

def perform_adf_test(returns: pd.Series) -> dict:
    """
    Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.
    
    Parameters:
        returns (pd.Series): Log returns series.
        
    Returns:
        dict: Summary of ADF test results.
    """
    clean_returns = returns.dropna()
    result = adfuller(clean_returns)
    
    return {
        "test_statistic": float(result[0]),
        "p_value": float(result[1]),
        "lags_used": int(result[2]),
        "n_obs": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "is_stationary": bool(result[1] < 0.05)
    }

def calculate_rolling_volatility(returns: pd.Series, window: int = 21) -> pd.DataFrame:
    """
    Calculates rolling historical volatility (standard deviation of daily log returns).
    Returns both daily and annualized volatility series.
    
    Annualized Volatility = Daily Volatility * sqrt(252)
    
    Parameters:
        returns (pd.Series): Log returns series.
        window (int): Size of rolling window in trading days (default 21 days ~ 1 trading month).
        
    Returns:
        pd.DataFrame: DataFrame containing 'Daily' and 'Annualized' rolling volatility.
    """
    rolling_daily = returns.rolling(window=window).std()
    rolling_annualized = rolling_daily * np.sqrt(252)
    
    df_vol = pd.DataFrame({
        "Daily": rolling_daily,
        "Annualized": rolling_annualized
    }, index=returns.index)
    
    return df_vol.dropna()

def get_descriptive_stats(returns: pd.Series) -> dict:
    """
    Calculates descriptive and annual statistical metrics for stock returns.
    
    Parameters:
        returns (pd.Series): Log returns series.
        
    Returns:
        dict: A dictionary of calculated statistics.
    """
    daily_mean = float(returns.mean())
    daily_std = float(returns.std())
    
    return {
        "daily_mean": daily_mean,
        "daily_std": daily_std,
        "annualized_mean": daily_mean * 252,
        "annualized_std": daily_std * np.sqrt(252),
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurt()),
        "min": float(returns.min()),
        "max": float(returns.max())
    }
