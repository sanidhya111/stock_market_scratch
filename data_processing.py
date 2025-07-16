import ast
import pandas as pd


# Processing Stock DATA
def processed_stock_data(raw_stock_df):
    stock_time_series_df = raw_stock_df['Time Series (Daily)'][5:]
    stock_name_frm_df = raw_stock_df['Meta Data'][1]
    stock_date_frm_df = raw_stock_df['Meta Data'][2]

    # Extract keys of the dictionary series for imported stock data

    dict_series = stock_time_series_df.apply(lambda x:ast.literal_eval(x) if isinstance(x, str) else x)
    all_keys = dict_series.apply(lambda x: set(x.keys()))
    master_keys = set().union(*all_keys)
    sorted_master_keys = sorted(master_keys)

    stock_processed_data = pd.DataFrame({key: stock_time_series_df.apply(lambda x:ast.literal_eval(x)[key] if isinstance(x, str) else x[key])
                                   for key in sorted_master_keys})

    stock_processed_data = stock_processed_data.astype(float)
    stock_processed_data.index = range(1, len(stock_processed_data)+1)
    return stock_processed_data

# END