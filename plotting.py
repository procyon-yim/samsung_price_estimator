import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from scipy.interpolate import make_interp_spline

def plot_prediction(history_df, prediction, ticker):
    """
    Plots historical stock prices and the predicted future price.
    
    Args:
        history_df (pd.DataFrame): DataFrame containing historical stock data. 
                                   Must have 'Date' and 'Close' columns (or 'Close' with Date index).
        prediction (float): Predicted stock price.
        ticker (str): Stock ticker symbol for the title.
        
    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    
    # Plotting with Seaborn
    sns.set_theme(style="white", context="talk") # Cleaner "talk" context for larger fonts, white background
    
    # Create a dataframe for plotting
    plot_df = history_df.reset_index()
    # Assuming index is date
    plot_df.columns = ['Date', 'Close'] if 'Date' not in plot_df.columns else plot_df.columns
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. Historical Data - Smooth line, nice color
    # Use spline interpolation for smoother lines
    try:
        # Convert dates to numbers for interpolation
        # Ensure dates are datetime
        plot_df['Date'] = pd.to_datetime(plot_df['Date'])
        date_nums = pd.to_numeric(plot_df['Date'])
        
        # Create smooth x axis
        date_nums_smooth = np.linspace(date_nums.min(), date_nums.max(), 300)
        
        # Spline interpolation
        spl = make_interp_spline(date_nums, plot_df['Close'], k=3)
        close_smooth = spl(date_nums_smooth)
        
        # Convert back to dates
        dates_smooth = pd.to_datetime(date_nums_smooth)
        
        # Plot smooth line
        sns.lineplot(x=dates_smooth, y=close_smooth, ax=ax, color='#4c72b0', linewidth=2.5, label='Historical')
        
        # Plot original points as scatter to show data
        sns.scatterplot(data=plot_df, x='Date', y='Close', ax=ax, color='#4c72b0', s=60, marker='o')
        
    except ImportError:
        # Fallback if scipy is not available
        sns.lineplot(data=plot_df, x='Date', y='Close', ax=ax, label='Historical', color='#4c72b0', linewidth=2.5, marker='o', markersize=8)
    except Exception as e:
         # Fallback for other errors during smoothing
         sns.lineplot(data=plot_df, x='Date', y='Close', ax=ax, label='Historical', color='#4c72b0', linewidth=2.5, marker='o', markersize=8)

    
    # Add prediction point
    last_date = plot_df['Date'].iloc[-1]
    try:
        next_date = last_date + datetime.timedelta(days=1)
    except:
        last_date = pd.to_datetime(last_date)
        next_date = last_date + datetime.timedelta(days=1)

    # 2. Connection Line - Dashed, grey, subtle
    connector_df = pd.DataFrame({
        'Date': [last_date, next_date],
        'Close': [plot_df['Close'].iloc[-1], prediction]
    })
    sns.lineplot(data=connector_df, x='Date', y='Close', ax=ax, color='grey', linestyle='--', linewidth=2, label='예측 흐름')

    # 3. Prediction Point (No Uncertainty displayed)
    # Plot prediction point
    ax.scatter([next_date], [prediction], color='#c44e52', s=100, zorder=5, label='예측')
    
    # Add simple text annotation (Prediction only)
    text_str = f"{prediction:,.0f} KRW"
    ax.text(next_date, prediction * 1.002, text_str, ha='center', va='bottom', fontsize=12, color='#c44e52', fontweight='bold')

    # Polish Layout
    # Simple Title
    ax.set_title(f"{ticker} Price Prediction", fontsize=20, fontweight='bold', pad=20) 
    
    # Remove unnecessary labels
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    # Remove top and right spines
    sns.despine(trim=True, offset=10)
    
    # Format X-axis dates cleaner
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=0, fontsize=12) # Horizontal dates if possible, or slight rotation
    plt.yticks(fontsize=12)
    
    # Remove legend for simplicity (Annotation explains the prediction)
    if ax.get_legend():
        ax.get_legend().remove()
    
    ax.grid(axis='y', linestyle='--', alpha=0.3) 
    
    return fig
