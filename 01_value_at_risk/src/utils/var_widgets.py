"""
Interactive widgets for Value at Risk analysis
Provides ticker selection and date picker widgets for Jupyter notebooks
"""

import ipywidgets as widgets
from IPython.display import display
from datetime import datetime, timedelta

# Tickers dictionary
tickers = {
    "^GSPC": "S&P 500",
    "BTC-USD": "Bitcoin",
    "CLP=X": "USD/CLP (Chile Peso)",
    "SQM": "SQM (Lithium)",
    "HG=F": "Copper Futures",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "CL=F": "Crude Oil",
    "TLT": "US Treasuries (20Y)",
    "VXX": "VIX Volatility"
}


def create_ticker_widget():
    """
    Create a dropdown widget for ticker selection
    
    Returns:
        ipywidgets.Dropdown: Dropdown widget with ticker options
    """
    ticker_widget = widgets.Dropdown(
        options=[(f"{name} ({symbol})", symbol) for symbol, name in tickers.items()],
        value="BTC-USD",
        description='Ticker:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )
    return ticker_widget


def create_date_widget(default_date=None):
    """
    Create a date picker widget
    
    Args:
        default_date (str, optional): Default date in 'YYYY-MM-DD' format. 
                                     Defaults to 2019-01-01 if None.
    
    Returns:
        ipywidgets.DatePicker: Date picker widget
    """
    if default_date is None:
        default_date = datetime(2019, 1, 1)
    elif isinstance(default_date, str):
        default_date = datetime.strptime(default_date, '%Y-%m-%d')
    
    date_widget = widgets.DatePicker(
        description='Report Date:',
        value=default_date,
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )
    return date_widget


def create_var_widgets(default_ticker="BTC-USD", default_date="2019-01-01"):
    """
    Create a complete widget interface for VaR analysis
    
    Args:
        default_ticker (str): Default ticker symbol
        default_date (str): Default date in 'YYYY-MM-DD' format
    
    Returns:
        tuple: (ticker_widget, date_widget, output_widget)
    """
    # Create widgets
    ticker_widget = create_ticker_widget()
    ticker_widget.value = default_ticker
    
    date_widget = create_date_widget(default_date)
    
    # Create output widget to display selections
    output = widgets.Output()
    
    # Create update function
    def on_value_change(change):
        with output:
            output.clear_output()
            print(f"Selected Ticker: {ticker_widget.value} ({tickers[ticker_widget.value]})")
            print(f"Selected Date: {date_widget.value}")
    
    # Attach observers
    ticker_widget.observe(on_value_change, names='value')
    date_widget.observe(on_value_change, names='value')
    
    # Display initial selection
    with output:
        print(f"Selected Ticker: {ticker_widget.value} ({tickers[ticker_widget.value]})")
        print(f"Selected Date: {date_widget.value}")
    
    # Create container
    container = widgets.VBox([
        widgets.HTML("<h3>VaR Analysis Parameters</h3>"),
        ticker_widget,
        date_widget,
        output
    ])
    
    return ticker_widget, date_widget, container


def display_var_widgets(default_ticker="BTC-USD", default_date="2019-01-01"):
    """
    Display the VaR widgets and return the widget objects
    
    Args:
        default_ticker (str): Default ticker symbol
        default_date (str): Default date in 'YYYY-MM-DD' format
    
    Returns:
        tuple: (ticker_widget, date_widget)
    
    Example:
        ticker_widget, date_widget = display_var_widgets()
        # Access selected values:
        selected_ticker = ticker_widget.value
        selected_date = date_widget.value.strftime('%Y-%m-%d')
    """
    ticker_widget, date_widget, container = create_var_widgets(default_ticker, default_date)
    display(container)
    return ticker_widget, date_widget


# Example usage in notebook:
if __name__ == "__main__":
    print("Import this module in your Jupyter notebook:")
    print("from var_widgets import display_var_widgets")
    print("\nticker_widget, date_widget = display_var_widgets()")
    print("\n# Access values:")
    print("ticker = ticker_widget.value")
    print("date_rep = date_widget.value.strftime('%Y-%m-%d')")
