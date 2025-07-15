import pandas as pd
import os
from my_api_keys import *
import requests
import re # to remove all the special characters from the user_input
from io import StringIO # Correcting import of stock list data
import datetime


# from stock_prediction_app import data_frm_streamlit

# selected_stock, selected_stock_data, selected_name = data_frm_streamlit(raw_stock_list())

# user_input= input("Please enter the stock you want to download: ")
# user_input = "RELIANCE.BSE" # For coding purpose only

# user_input = selected_name # Work on converting the code into functions wherever called
# stock_name = user_input.lower()
# stock_name = re.sub(r"\W+", "_", stock_name) # remove all the special characters and replace with _

stock_list_url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={alpha_vantage_api_key}"
# stock_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={user_input}&outputsize=full&apikey={alpha_vantage_api_key}"


today = datetime.date.today()

# stock_filepath = f"raw_df_{stock_name}.csv"
stock_list_filepath = f"stock_list_{today}.csv"

# refresh_stocklist = input("Do you want FRESH stocklist data? (y/n): ").lower()
refresh_stocklist = 'n' # For coding purpose only
# refresh_stock_data = input("Do you want FRESH stock data? (y/n): ").lower()
# refresh_stock_data = 'y' # For coding purpose only


def raw_stock_list():
    if refresh_stocklist == "n" and os.path.exists(stock_list_filepath):
        stock_list_df = pd.read_csv(stock_list_filepath)
        print(f"\nNote: Using local stock list file {stock_list_filepath}\n")
    else:
        print(f"\nNote: Stock list file missing or refresh requested. Downloading from Alpha Vantage...\n")
        response = requests.get(stock_list_url)
        decoded = response.content.decode("utf-8")
        stock_list = pd.read_csv(StringIO(decoded))
        stock_list_df = pd.DataFrame(stock_list)
        stock_list_df.to_csv(stock_list_filepath, index=False)

    return stock_list_df


def raw_stock_data(user_input: str, refresh_stock_data: str = "y"):
    stock_name = user_input.lower()
    stock_name = re.sub(r"\W+", "_", stock_name)

    stock_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={user_input}&outputsize=full&apikey={alpha_vantage_api_key}"
    stock_filepath = f"raw_df_{stock_name}.csv"

    if refresh_stock_data == "n" and os.path.isfile(stock_filepath):
        print(f"\nNote: {stock_filepath} exists, skipping stock data download from Alpha Vantage.\n")
        data = pd.read_csv(stock_filepath)
        df = pd.DataFrame(data)

    else:
        response = requests.get(stock_url)
        data = response.json()
        df = pd.DataFrame(data)
        df.to_csv(f"raw_df_{stock_name}.csv", index=False)
        print(f"\nNote: Downloaded fresh stock data file {stock_filepath}\n")
    return df

stock_df = raw_stock_data(selected_name)

if __name__ == "__main__":
    # print(raw_stock_data(stock_url, stock_filepath))
    raw_imported_data = raw_stock_data()

    print("\n")
    print("Below is the info on the stock list data:")
    raw_imported_data.info()
    print(raw_imported_data.describe())
    print("\n")

    # print(raw_stock_list(stock_list_url, stock_list_filepath))
    raw_imported_stocklist = raw_stock_list()

    print("\n")
    print("Below is the info on the stock list data:")
    raw_imported_stocklist.info()
    print(raw_imported_stocklist.describe())
    print("\n")

