import torch
import pandas as pd

def create_features(df, alpha=0.05, rolling=22, features=['log_ret']):
    """
    Prepares data specifically for the Single Neuron test.
    X = Recent Variance (Last 5 days)
    y = Actual Next Day Return
    """
    df_features = df.copy()

    columns = []
    # features=['std', 'mean', 'log_ret', 'variance', 'historical_var', 'historical_var_2', 'skewness', 'kurtosis']
    # features = ['log_ret', 'std', 'mean', 'variance', 'historical_var', 'historical_var_2', 'skewness', 'kurtosis']

    if 'log_ret' in features:
        columns.append(f'log_ret')

    if 'std' in features:
        df_features[f'std_{rolling}d'] = df_features['log_ret'].rolling(rolling).std()
        columns.append(f'std_{rolling}d')
    if 'mean' in features:
        df_features[f'mean_{rolling}d'] = df_features['log_ret'].rolling(rolling).mean()
        columns.append(f'mean_{rolling}d')
    if 'log_ret^2' in features:
        df_features[f'log_ret^2'] = df_features['log_ret']**2
        columns.append(f'log_ret^2')
    if 'variance' in features:
        df_features[f'variance_{rolling}d'] = df_features['log_ret'].rolling(rolling).var()
        columns.append(f'variance_{rolling}d')
    if 'historical_var' in features:
        df_features[f'historical_var_{rolling}d'] = df_features['log_ret'].rolling(rolling).quantile(alpha)
        columns.append(f'historical_var_{rolling}d')
    if 'historical_var_2' in features:
        df_features[f'historical_var_2_{rolling}d'] = df_features['log_ret'].rolling(rolling).quantile(1-alpha)
        columns.append(f'historical_var_2_{rolling}d')
    
    if 'skewness' in features:
        df_features[f'skewness_{rolling}d'] = df_features['log_ret'].rolling(rolling).skew()
        columns.append(f'skewness_{rolling}d')
    if 'kurtosis' in features:
        df_features[f'kurtosis_{rolling}d'] = df_features['log_ret'].rolling(rolling).kurt()
        columns.append(f'kurtosis_{rolling}d')
    
    # Target: The ACTUAL return of the next day (shifted back)
    # We want to predict tomorrow's return distribution based on today's variance
    df_features['target_return'] = df_features['log_ret'].shift(-1)
    df_features.dropna(inplace=True)
    
    # Convert to Tensors
    X = torch.tensor(df_features[columns].values, dtype=torch.float32)
    y = torch.tensor(df_features[['target_return']].values, dtype=torch.float32)
    
    dates = pd.Series([pd.to_datetime(date) for date in df_features.index])
    data = {
        'dates': dates,
        'y': y,
        'X': X
    }
    
    return data