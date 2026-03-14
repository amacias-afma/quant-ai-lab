from datetime import date, timedelta
import pandas as pd
import yfinance as yf


def fetch_asset_data(
    ticker: str = "BTC-USD",
    end_date: date = None,
    tenor: str = "5Y"
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol. Default is 'BTC-USD' (Bitcoin).
    end_date : date
        End date for the historical data. Default is yesterday.
    tenor : str
        Historical data period. E.g. '5Y', '2Y', '1Y', '6mo'. Default is '5Y'.

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV columns indexed by date.
    """
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    # Convert tenor to a start date
    tenor_map = {
        "1Y": 365, "2Y": 730, "3Y": 1095,
        "5Y": 1825, "10Y": 3650
    }
    if tenor in tenor_map:
        start_date = end_date - timedelta(days=tenor_map[tenor])
    else:
        raise ValueError(f"Unsupported tenor '{tenor}'. Use one of: {list(tenor_map.keys())}")

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df.columns = df.columns.get_level_values(0)  # flatten MultiIndex if present
    return df
