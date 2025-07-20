import pandas as pd
import re
import glob
import os
import datetime



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

# Date extraction from the existing files
download_dir = "downloaded_data"

# 📌 Utility function for file date extraction (Place this here!)
def get_latest_stock_file_date(symbol):
    stock_name = re.sub(r"\W+", "_", symbol.lower())
    pattern = os.path.join(download_dir, f"raw_df_{stock_name}_*.csv")
    matching_files = glob.glob(pattern)

    def extract_date(filename):
        try:
            parts = filename.replace(".csv", "").split("_")
            date_str = parts[-1]
            return datetime.datetime.strptime(date_str, "%d-%m-%Y")
        except Exception:
            return None

    dated_files = [(f, extract_date(f)) for f in matching_files if extract_date(f)]
    if dated_files:
        latest_file, latest_date = sorted(dated_files, key=lambda x: x[1])[-1]
        return latest_file, latest_date.strftime("%d-%m-%Y")
    else:
        return None, "Not available"

# END of date extraction code


# Function to get column wise adata list, column names and dates for all the data
def label_frm_columns(processed_stock_df):
    # Extract target columns from DataFrame
    stock_columns_list = processed_stock_df.columns

    # Prepare data and labels
    datas = [processed_stock_df[col] for col in stock_columns_list]
    column_names = [f"📈 {col.capitalize()} Plot" for col in stock_columns_list]
    dates = processed_stock_df['date']  # Optional, used if present

    null_count_dict = {}
    for name, series in zip(column_names, datas):
        null_count = series.isnull().sum()
        if null_count > 0:
            print(f"🚫 {name} has {null_count} missing values.")
            null_count_dict[name] = null_count
        else:
            print(f"✅ {name} is complete — no missing values.")
            null_count_dict[name] = 0

    return datas, column_names, dates, null_count_dict

# Below code returns data as a series
def sanitized_data(processed_stock_data):
    # data has some missing values, replacing it mean value and skipping the data column
    sanitized_datas, column_names, dates,_ = label_frm_columns(processed_stock_data)

    for i, data in enumerate(sanitized_datas[1:]):
        filled = data.fillna(data.mean())
        sanitized_datas[i + 1] = filled

    null_count_dict = {}
    for name, series in zip(column_names, sanitized_datas):
        null_count = series.isnull().sum()
        if null_count > 0:
            print(f"🚫 {name} has {null_count} missing values.")
            null_count_dict[name] = null_count
        else:
            print(f"✅ {name} is complete — no missing values.")
            null_count_dict[name] = series.count()

    return sanitized_datas, null_count_dict

# To use the sanitized data in main.py Dataframe is required
def sanitized_data_df(processed_stock_df):
    sanitized_datas, column_names, dates, _ = label_frm_columns(processed_stock_df)

    for i, data in enumerate(sanitized_datas[1:]):
        filled = data.fillna(data.mean())
        sanitized_datas[i + 1] = filled

    # Reconstruct DataFrame
    sanitized_df = pd.DataFrame({
        col.replace("📈 ", "").replace(" Plot", "").lower(): series
        for col, series in zip(column_names, sanitized_datas)
    })

    sanitized_df['date'] = dates
    sanitized_df = sanitized_df[['date'] + [col for col in sanitized_df.columns if col != 'date']]

    print("\nPrinting the data after sanitizing...\n")
    for name, series in zip(column_names, sanitized_datas):
        null_count = series.isnull().sum()
        if null_count > 0:
            print(f"🚫 {name} has {null_count} missing values.")

        else:
            print(f"✅ {name} is complete — no missing values.")

    return sanitized_df

if __name__ == '__main__':
    from fetch_n_save_data import raw_stock_data
    raw_stock_df = raw_stock_data(user_input='A', refresh_stock_data='n')
    processed_stock_data = processed_stock_data(raw_stock_df)
    processed_stock_data.info()
    print(f'Using describe below: \n{processed_stock_data.describe()}')

    # data has some missing values, replacing it mean value and skipping the data column
    datas, column_names, dates = label_frm_columns(processed_stock_data)

    for i, data in enumerate(datas[1:]):
        filled = data.fillna(data.mean())
        datas[i + 1] = filled


    print(datas)
    for name, series in zip(column_names, datas):
        null_count = series.isnull().sum()
        if null_count > 0:
            print(f"🚫 {name} has {null_count} missing values.")
        else:
            print(f"✅ {name} is complete — no missing values.")