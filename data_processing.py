import pandas as pd
import numpy as np  # ADDED: Missing import for infinity handling
import re
import glob
import os
import datetime


def remove_non_finite(df, cols):
    """
    ADDED: Reusable function to handle infinity and extreme values
    Replace ±inf with NaN, then apply forward/backward fill strategy

    Parameters:
    -----------
    df : pandas.DataFrame
    cols : list of str - columns to sanitize

    Returns:
    --------
    pandas.DataFrame with infinite values handled
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()

    # Replace infinite values with NaN
    df_clean[cols] = df_clean[cols].replace([np.inf, -np.inf], np.nan)

    # Apply forward fill, then backward fill, finally use mean if still NaN
    for col in cols:
        if col in df_clean.columns:
            original_nulls = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')

            # If still has NaN after forward/backward fill, use mean
            if df_clean[col].isnull().any():
                mean_val = df_clean[col].mean()
                if not pd.isna(mean_val):
                    df_clean[col] = df_clean[col].fillna(mean_val)

            # Optional: clip extreme values to prevent float32 overflow
            if df_clean[col].dtype in [np.float32, np.float64]:
                df_clean[col] = df_clean[col].clip(lower=-1e6, upper=1e6)

    # Drop any rows that still have NaN in the specified columns
    before_drop = len(df_clean)
    df_clean = df_clean.dropna(subset=cols)
    after_drop = len(df_clean)

    if before_drop != after_drop:
        print(f"🗑️ Removed {before_drop - after_drop} rows due to remaining invalid values")

    return df_clean


# Processing Stock DATA
def processed_stock_data(raw_stock_df):
    # ENHANCED: Added validation
    if raw_stock_df.empty:
        print("⚠️ Warning: Empty DataFrame provided to processed_stock_data")
        return pd.DataFrame()

    df = raw_stock_df.copy()

    # Sanitize column names (in case future sources introduce messy formatting)
    df.columns = [col.lower().strip() for col in df.columns]

    # Ensure date column is datetime-type
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # ENHANCED: Handle conversion errors
    else:
        print("⚠️ Warning: No 'date' column found in processed_stock_data")

    # Drop any non-numeric metadata if needed (like 'volume', 'symbol') — optional
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    processed_data = df[['date'] + numeric_cols] if 'date' in df.columns else df[numeric_cols]

    # Example: sort by date and compute indicators
    if 'date' in processed_data.columns:
        processed_data = processed_data.sort_values(by='date')

    if 'close' in processed_data.columns:
        # ENHANCED: Add minimum periods to handle edge cases
        processed_data['sma_5'] = processed_data['close'].rolling(window=5, min_periods=1).mean()
        processed_data['sma_10'] = processed_data['close'].rolling(window=10, min_periods=1).mean()

    # Reset index for Streamlit-friendly display
    processed_data = processed_data.reset_index(drop=True)

    return processed_data


# END

# Date extraction from the existing files
download_dir = "downloaded_data"


# 📌 Utility function for file date extraction (Place this here!)
def get_latest_stock_file_date(symbol):
    # ENHANCED: Added validation
    if not symbol:
        print("⚠️ Warning: No symbol provided to get_latest_stock_file_date")
        return None, "No symbol provided"

    stock_name = re.sub(r"\W+", "_", symbol.lower())
    pattern = os.path.join(download_dir, f"raw_df_{stock_name}_*.csv")
    matching_files = glob.glob(pattern)

    def extract_date(filename):
        try:
            parts = filename.replace(".csv", "").split("_")
            date_str = parts[-1]
            return datetime.datetime.strptime(date_str, "%d-%m-%Y")
        except (ValueError, IndexError) as e:  # ENHANCED: More specific error handling
            print(f"⚠️ Warning: Could not extract date from {filename}: {e}")
            return None

    dated_files = [(f, extract_date(f)) for f in matching_files if extract_date(f)]
    if dated_files:
        latest_file, latest_date = sorted(dated_files, key=lambda x: x[1])[-1]
        return latest_file, latest_date.strftime("%d-%m-%Y")
    else:
        return None, "Not available"


# END of date extraction code


# Function to get column wise data list, column names and dates for all the data
def label_frm_columns(processed_stock_df):
    # ENHANCED: Added validation
    if processed_stock_df.empty:
        print("⚠️ Warning: Empty DataFrame provided to label_frm_columns")
        return [], [], pd.Series(dtype='datetime64[ns]'), {}

    # Extract target columns from DataFrame
    stock_columns_list = processed_stock_df.columns

    # Prepare data and labels
    datas = [processed_stock_df[col] for col in stock_columns_list]
    column_names = [f"📈 {col.capitalize()} Plot" for col in stock_columns_list]
    dates = processed_stock_df['date'] if 'date' in processed_stock_df.columns else pd.Series(
        dtype='datetime64[ns]')  # ENHANCED: Handle missing date

    null_count_dict = {}
    for name, series in zip(column_names, datas):
        null_count = series.isnull().sum()
        # ENHANCED: Also check for infinite values
        inf_count = np.isinf(series).sum() if pd.api.types.is_numeric_dtype(series) else 0

        if null_count > 0:
            print(f"🚫 {name} has {null_count} missing values.")
            null_count_dict[name] = null_count
        elif inf_count > 0:
            print(f"⚠️ {name} has {inf_count} infinite values.")
            null_count_dict[name] = f"{inf_count} inf values"
        else:
            print(f"✅ {name} is complete — no missing values.")
            null_count_dict[name] = 0

    return datas, column_names, dates, null_count_dict


# Below code returns data as a series
def sanitized_data(processed_stock_data):
    # ENHANCED: Added validation
    if processed_stock_data.empty:
        print("⚠️ Warning: Empty DataFrame provided to sanitized_data")
        return [], {}

    # data has some missing values, replacing it mean value and skipping the data column
    sanitized_datas, column_names, dates, _ = label_frm_columns(processed_stock_data)

    # ENHANCED: Better handling of non-numeric columns and infinite values
    for i, data in enumerate(sanitized_datas[1:]):
        # Skip if not numeric
        if not pd.api.types.is_numeric_dtype(data):
            continue

        # Handle infinite values first, then NaN values
        data_clean = data.replace([np.inf, -np.inf], np.nan)

        # Enhanced imputation strategy
        if data_clean.isnull().any():
            mean_val = data_clean.mean()
            if pd.isna(mean_val):
                # If mean is also NaN, use forward/backward fill
                filled = data_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
            else:
                filled = data_clean.fillna(mean_val)
            sanitized_datas[i + 1] = filled
        else:
            sanitized_datas[i + 1] = data_clean

    null_count_dict = {}
    for name, series in zip(column_names, sanitized_datas):
        null_count = series.isnull().sum()
        # ENHANCED: Also report infinite values in final check
        inf_count = np.isinf(series).sum() if pd.api.types.is_numeric_dtype(series) else 0

        if null_count > 0:
            print(f"🚫 {name} has {null_count} missing values.")
            null_count_dict[name] = null_count
        elif inf_count > 0:
            print(f"⚠️ {name} still has {inf_count} infinite values.")
            null_count_dict[name] = f"{inf_count} inf values"
        else:
            print(f"✅ {name} is complete — no missing values.")
            null_count_dict[name] = series.count()

    return sanitized_datas, null_count_dict


# To use the sanitized data in main.py Dataframe is required
def sanitized_data_df(processed_stock_df):
    # ENHANCED: Added validation
    if processed_stock_df.empty:
        print("⚠️ Warning: Empty DataFrame provided to sanitized_data_df")
        return pd.DataFrame()

    try:
        sanitized_datas, column_names, dates, _ = label_frm_columns(processed_stock_df)

        # ENHANCED: More robust column reconstruction
        column_data = {}

        for i, data in enumerate(sanitized_datas[1:]):  # Skip first column (assumed to be date)
            col_name = column_names[i + 1].replace("📈 ", "").replace(" Plot", "").lower()

            # Handle infinite values first, then NaN values
            if pd.api.types.is_numeric_dtype(data):
                data_clean = data.replace([np.inf, -np.inf], np.nan)

                # Enhanced imputation strategy
                if data_clean.isnull().any():
                    mean_val = data_clean.mean()
                    if pd.isna(mean_val):
                        # If mean is also NaN, use forward/backward fill
                        filled = data_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
                    else:
                        filled = data_clean.fillna(mean_val)
                    column_data[col_name] = filled
                else:
                    column_data[col_name] = data_clean
            else:
                column_data[col_name] = data

        # Reconstruct DataFrame
        sanitized_df = pd.DataFrame(column_data)

        # Handle date column separately
        if not dates.empty:
            sanitized_df['date'] = dates
            sanitized_df = sanitized_df[['date'] + [col for col in sanitized_df.columns if col != 'date']]
        elif 'date' in processed_stock_df.columns:
            # Fallback: get date from original
            sanitized_df['date'] = processed_stock_df['date']
            sanitized_df = sanitized_df[['date'] + [col for col in sanitized_df.columns if col != 'date']]

        # CRITICAL ENHANCEMENT: Final cleanup of any remaining infinite values
        numeric_cols = sanitized_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Check if there are any infinite values left
            total_inf = np.isinf(sanitized_df[numeric_cols]).sum().sum()
            if total_inf > 0:
                print(f"🔧 Applying final cleanup for {total_inf} remaining infinite values...")
                sanitized_df = remove_non_finite(sanitized_df, numeric_cols)

        print("\nPrinting the data after sanitizing...\n")
        for name, series in zip(column_names, sanitized_datas):
            # Get corresponding column name in sanitized_df
            clean_col_name = name.replace("📈 ", "").replace(" Plot", "").lower()

            if clean_col_name in sanitized_df.columns:
                final_series = sanitized_df[clean_col_name]
                null_count = final_series.isnull().sum()
                inf_count = np.isinf(final_series).sum() if pd.api.types.is_numeric_dtype(final_series) else 0

                if null_count > 0:
                    print(f"🚫 {name} has {null_count} missing values.")
                elif inf_count > 0:
                    print(f"⚠️ {name} still has {inf_count} infinite values.")
                else:
                    print(f"✅ {name} is complete — no missing values.")
            else:
                # Fallback for columns that might not have made it to final df
                null_count = series.isnull().sum()
                inf_count = np.isinf(series).sum() if pd.api.types.is_numeric_dtype(series) else 0

                if null_count > 0:
                    print(f"🚫 {name} has {null_count} missing values.")
                elif inf_count > 0:
                    print(f"⚠️ {name} has {inf_count} infinite values.")
                else:
                    print(f"✅ {name} is complete — no missing values.")

        return sanitized_df

    except Exception as e:
        print(f"❌ Error in sanitized_data_df: {str(e)}")
        print("⚠️ Returning original processed DataFrame as fallback")
        return processed_stock_df


def validate_data_for_ai(df, feature_cols):
    """
    NEW: Validate data specifically for AI model training
    Returns detailed report of data quality issues
    """
    if df.empty:
        return {"status": "error", "message": "Empty DataFrame"}

    report = {
        "status": "ok",
        "total_rows": len(df),
        "issues": [],
        "warnings": []
    }

    for col in feature_cols:
        if col not in df.columns:
            report["issues"].append(f"Missing required column: {col}")
            continue

        series = df[col]
        null_count = series.isnull().sum()
        inf_count = np.isinf(series).sum() if pd.api.types.is_numeric_dtype(series) else 0

        if null_count > 0:
            report["warnings"].append(f"{col}: {null_count} null values")
        if inf_count > 0:
            report["issues"].append(f"{col}: {inf_count} infinite values")

    if report["issues"]:
        report["status"] = "error"
    elif report["warnings"]:
        report["status"] = "warning"

    return report


if __name__ == '__main__':
    from fetch_n_save_data import raw_stock_data

    print("🧪 Testing enhanced data_processing.py...\n")

    raw_stock_df = raw_stock_data(user_input='A', refresh_stock_data='n')

    if not raw_stock_df.empty:
        print("📊 Processing stock data...")
        processed_stock_data_result = processed_stock_data(raw_stock_df)
        processed_stock_data_result.info()
        print(f'Using describe below: \n{processed_stock_data_result.describe()}')

        print("\n📋 Getting column labels and data...")
        # data has some missing values, replacing it mean value and skipping the data column
        datas, column_names, dates, null_count_dict = label_frm_columns(processed_stock_data_result)

        print(f"\n📈 Null count summary: {null_count_dict}")

        print("\n🧹 Applying sanitization to series data...")
        for i, data in enumerate(datas[1:]):
            original_nulls = data.isnull().sum()
            original_infs = np.isinf(data).sum() if pd.api.types.is_numeric_dtype(data) else 0

            # Apply same logic as sanitized_data function
            data_clean = data.replace([np.inf, -np.inf], np.nan) if pd.api.types.is_numeric_dtype(data) else data

            if data_clean.isnull().any():
                mean_val = data_clean.mean() if pd.api.types.is_numeric_dtype(data_clean) else None
                if pd.isna(mean_val):
                    filled = data_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
                else:
                    filled = data_clean.fillna(mean_val)
                datas[i + 1] = filled
            else:
                datas[i + 1] = data_clean

            final_nulls = datas[i + 1].isnull().sum()
            final_infs = np.isinf(datas[i + 1]).sum() if pd.api.types.is_numeric_dtype(datas[i + 1]) else 0

            col_name = column_names[i + 1]
            print(f"🔄 {col_name}: {original_nulls} null + {original_infs} inf → {final_nulls} null + {final_infs} inf")

        print("\n📊 Final series validation:")
        for name, series in zip(column_names, datas):
            null_count = series.isnull().sum()
            inf_count = np.isinf(series).sum() if pd.api.types.is_numeric_dtype(series) else 0

            if null_count > 0:
                print(f"🚫 {name} has {null_count} missing values.")
            elif inf_count > 0:
                print(f"⚠️ {name} has {inf_count} infinite values.")
            else:
                print(f"✅ {name} is complete — no missing values.")

        print("\n🗂️ Testing DataFrame sanitization...")
        sanitized_df = sanitized_data_df(processed_stock_data_result)

        print(f"\n📋 Final DataFrame shape: {sanitized_df.shape}")
        print(f"📋 Final DataFrame columns: {list(sanitized_df.columns)}")

        # Test AI validation
        test_features = ['close', 'sma_5', 'sma_10', 'volume']
        available_features = [col for col in test_features if col in sanitized_df.columns]

        if available_features:
            print(f"\n🤖 Testing AI data validation for features: {available_features}")
            ai_report = validate_data_for_ai(sanitized_df, available_features)
            print(f"🎯 AI Validation Report: {ai_report}")

        print("\n✅ Enhanced data_processing.py testing completed!")
    else:
        print("❌ No raw data available for testing")

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