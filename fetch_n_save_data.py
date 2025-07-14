import pandas as pd
import os
from my_api_keys import *
import requests
import re # to remove all the special characters from the user_input

# user_input= input("Please enter the stock you want to download: ")
user_input = "RELIANCE.BSE" # For coding purpose only
stock_name = user_input.lower()
stock_name = re.sub(r"\W+", "_", stock_name) # remove all the special characters and replace with _


stock_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={user_input}&outputsize=full&apikey={alpha_vantage_api_key}"
stock_list_url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={alpha_vantage_api_key}"

stock_filepath = f"raw_df_{stock_name}.csv"
stock_list_filepath = f"stock_list_{stock_name}.csv"

def raw_stock_list(stock_list_url, stock_list_filepath):
    if os.path.isfile(stock_list_filepath):
        stock_list_read = pd.read_csv(stock_list_filepath)
        stock_list_df = pd.DataFrame(stock_list_read)
        print(f"\nNote: Using local stock list file {stock_list_filepath}\n")

    else:
        response = requests.get(stock_list_url)
        # stock_list_json = response.json() # This line of code not required because this file comes as csv not json
        stock_list_df = pd.DataFrame(response)
        stock_list_df.to_csv(f"stock_list_{stock_name}.csv", index=False)
        print(f"\nNote: Downloaded fresh stock list file {stock_list_filepath}\n")

    return stock_list_df

def raw_stock_data(stock_url, stock_filepath):
    if os.path.isfile(stock_filepath):
        print(f"\nNote: {stock_filepath} exists, skipping stock data download from Alpha Vantage.\n")
        data = pd.read_csv(stock_filepath)
        df = pd.DataFrame(data)

    else:
        response = requests.get(stock_url)
        data = response.json()
        df = pd.DataFrame(data)
        df.to_csv(f"raw_df_{stock_name}.csv", index=False)
    return df

if __name__ == "__main__":
    # print(raw_stock_data(stock_url, stock_filepath))
    raw_imported_data = raw_stock_data(stock_url, stock_filepath)

    print("\n")
    print("Below is the info on the stock list data:")
    raw_imported_data.info()
    print(raw_imported_data.describe())
    print("\n")

    # print(raw_stock_list(stock_list_url, stock_list_filepath))
    raw_imported_stocklist = raw_stock_list(stock_list_url, stock_list_filepath)

    print("\n")
    print("Below is the info on the stock list data:")
    raw_imported_stocklist.info()
    print(raw_imported_stocklist.describe())
    print("\n")

