from datetime import date, timedelta
import pandas as pd
import yfinance as yf


def fetch_asset_data(
    ticker: str = "BTC-USD",
    start: date = None,
    end: date = None,
    # Legacy aliases kept for backward compatibility
    start_date: date = None,
    end_date: date = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol. Default is 'BTC-USD' (Bitcoin).
    start : date
        Start date for the historical data. Default is 5 years ago.
    end : date
        End date for the historical data. Default is yesterday.
    start_date : date
        Alias for `start` (kept for backward compatibility).
    end_date : date
        Alias for `end` (kept for backward compatibility).

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV columns indexed by date.
    """
    # Resolve aliases: 'start'/'end' take priority over 'start_date'/'end_date'
    resolved_start = start or start_date
    resolved_end   = end   or end_date

    # Apply defaults if still None
    if resolved_start is None:
        resolved_start = date.today() - timedelta(days=5 * 365)
    if resolved_end is None:
        resolved_end = date.today() - timedelta(days=1)

    df = yf.download(ticker, start=resolved_start, end=resolved_end, progress=False)
    df.columns = df.columns.get_level_values(0)  # flatten MultiIndex if present
    df = df[['Close']].rename(columns={'Close': 'prices'})

    return df
