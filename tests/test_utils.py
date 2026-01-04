import pandas as pd
import numpy as np
from src.utils.clean_data import clean_data

def test_clean_data_filling():
    """
    Test that clean_data correctly fills NaN values.
    """
    # Create a DataFrame with NaNs
    df = pd.DataFrame({
        'price': [100.0, np.nan, 102.0, 103.0]
    })
    
    # Run cleaning
    cleaned_df = clean_data(df)
    
    # Assert no NaNs remain
    assert not cleaned_df.isnull().values.any()
    # Forward fill verification: Index 1 should be 100.0
    assert cleaned_df.loc[1, 'price'] == 100.0

def test_clean_data_outliers():
    """
    Test that clean_data handles extreme outliers (if z-score logic is active).
    """
    # Create data with a massive spike
    df = pd.DataFrame({
        'price': [100.0, 101.0, 10000.0, 102.0, 103.0] 
    })
    
    # Depending on your implementation, this might print a warning or replace the value.
    # We just ensure the function runs without crashing on outliers.
    cleaned_df = clean_data(df)
    assert not cleaned_df.empty