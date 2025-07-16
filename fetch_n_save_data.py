import pandas as pd
import os
from my_api_keys import alpha_vantage_api_key
import requests
import re # to remove all the special characters from the user_input
from io import StringIO # Correcting import of stock list data
import datetime

today = datetime.date.today()
today = today.strftime('%d-%m-%Y')

# Define and ensure download folder exists
download_dir = 'downloaded_data'
os.makedirs(download_dir, exist_ok=True)


# ✅ FIX: Accept user_input as argument instead of hardcoding
def raw_stock_data(user_input, refresh_stock_data):
    stock_name = re.sub(r"\W+", "_", user_input.lower())
    stock_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={user_input}&outputsize=full&apikey={alpha_vantage_api_key}"
    stock_filepath = os.path.join(download_dir, f"raw_df_{stock_name}_{today}.csv")

    if refresh_stock_data == 'n' and os.path.exists(stock_filepath):
        print(f"\nNote: Using local stock data file {stock_filepath}\n")
        df = pd.read_csv(stock_filepath)
    else:
        print(f"\nNote: Downloading fresh stock data for {user_input}\n")
        response = requests.get(stock_url)
        data = response.json()
        df = pd.DataFrame(data)
        df.to_csv(stock_filepath, index=False)

    return df

def raw_stock_list(refresh_stocklist='n'):
    stock_list_url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={alpha_vantage_api_key}"
    stock_list_filepath = os.path.join(download_dir, f"stock_list_{today}.csv")

    # ✅ FIX: Add file existence check when refresh is 'n'
    if refresh_stocklist == "n" and os.path.exists(stock_list_filepath):
        stock_list_df = pd.read_csv(stock_list_filepath)
        print(f"\nNote: Using local stock list file {stock_list_filepath}\n")
    else:
        print(f"\nNote: Stock list missing or refresh requested. Downloading...\n")
        response = requests.get(stock_list_url)
        decoded = response.content.decode("utf-8")
        stock_list_df = pd.read_csv(StringIO(decoded))
        stock_list_df.to_csv(stock_list_filepath, index=False)

    return stock_list_df


if __name__ == "__main__":
    raw_imported_data = raw_stock_data(user_input="RELIANCE.BSE")

    print("\n")
    print("Below is the info on the stock list data:")
    raw_imported_data.info()
    print(raw_imported_data.describe())
    print("\n")

    print(raw_stock_list())
    raw_imported_stocklist = raw_stock_list()

    print("\n")
    print("Below is the info on the stock list data:")
    raw_imported_stocklist.info()
    print(raw_imported_stocklist.describe())
    print("\n")

