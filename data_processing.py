import ast
import pandas as pd


# Processing Stock DATA
def processed_stock_data(raw_stock_df):
    df = raw_stock_df.copy()

    # Sanitize column names (in case future sources introduce messy formatting)
    df.columns = [col.lower().strip() for col in df.columns]

    # Ensure date column is datetime-type
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Drop any non-numeric metadata if needed (like 'volume', 'symbol') — optional
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    processed_data = df[['date'] + numeric_cols] if 'date' in df.columns else df[numeric_cols]

    # Example: sort by date and compute indicators
    processed_data = processed_data.sort_values(by='date')
    if 'close' in processed_data.columns:
        processed_data['sma_5'] = processed_data['close'].rolling(window=5).mean()
        processed_data['sma_10'] = processed_data['close'].rolling(window=10).mean()

    # Reset index for Streamlit-friendly display
    processed_data = processed_data.reset_index(drop=True)

    return processed_data

# END