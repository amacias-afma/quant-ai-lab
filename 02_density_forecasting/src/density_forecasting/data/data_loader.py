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
    features: list = None
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
    df_final = df[['Close']].rename(columns={'Close': 'prices'})

    if 'volume' in features:
        df_final['volume'] = df['Volume']
    
    # print(df.columns)
    # df_final = df[['Close']].rename(columns={'Close': 'prices'})
    # if 'volume' in features:
    #     df['Volume'] = df['Volume'].replace(0, np.nan)
    #     df['Volume'] = df['Volume'].ffill().bfill()
    #     df_final['volume_ret'] = df['Volume'].pct_change().fillna(0)
        
        

    
    # # Compute log-returns for the models
    # df_final['returns'] = np.log(df_final['prices']).diff().dropna()

    # if 'std' in features:
    #     df_final['std'] = df_final['returns'].rolling(90).std()
    
    return df_final



def fetch_macro_features(start_date, end_date):
    # Download the external macro tickers
    macro_tickers = ['^TNX', 'DX-Y.NYB', 'HYG', 'HG=F']
    macro_df = yf.download(macro_tickers, start=start_date, end=end_date)['Close']
    
    # Forward fill any missing days (due to different market holidays)
    macro_df = macro_df.ffill().dropna()
    
    # 1. Rate Shock (TNX)
    macro_df['TNX_20_MA'] = macro_df['^TNX'].rolling(20).mean()
    macro_df['Macro_Rate_Shock'] = macro_df['^TNX'] / macro_df['TNX_20_MA']
    
    # 2. Dollar Momentum
    macro_df['DXY_10_MA'] = macro_df['DX-Y.NYB'].rolling(10).mean()
    macro_df['DXY_50_MA'] = macro_df['DX-Y.NYB'].rolling(50).mean()
    macro_df['Macro_USD_Trend'] = macro_df['DXY_10_MA'] / macro_df['DXY_50_MA']
    
    # 3. Credit Stress
    macro_df['Macro_Credit_Stress'] = macro_df['HYG'].pct_change(5)
    
    return macro_df[['Macro_Rate_Shock', 'Macro_USD_Trend', 'Macro_Credit_Stress']].dropna()

# In your main script:
# 1. Fetch macro data once
# macro_features = fetch_macro_features(start_date, end_date)
# 2. Merge it with your specific ticker's dataframe using the Date index
# df_features = df_features.join(macro_features, how='left').ffill().dropna()