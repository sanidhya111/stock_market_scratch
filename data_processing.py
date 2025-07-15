import ast
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import dtype
from tabulate import tabulate
from fetch_n_save_data import raw_stock_list, raw_stock_data
from stock_prediction_app import data_frm_streamlit

# Function calls
raw_stock_list_df = raw_stock_list()
selected_stock, selected_stock_data, selected_name = data_frm_streamlit()
raw_stock_df = raw_stock_data()


raw_stock_list_df.index = range(1, len(raw_stock_list_df)+1)

# Processing of Stock List DATA, not needed. As the imported data is already processed

column_names = raw_stock_list_df.columns
column_names = column_names.astype(str).tolist()


# Processing Stock DATA
def processed_stock_data(raw_stock_list_df):
    stock_time_series_df = raw_stock_df['Time Series (Daily)'][5:]
    # print(stock_time_series_df)

    stock_name_frm_df = raw_stock_df['Meta Data'][1]
    # print(stock_name_frm_df)

    stock_date_frm_df = raw_stock_df['Meta Data'][2]
    # print(stock_date_frm_df)

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