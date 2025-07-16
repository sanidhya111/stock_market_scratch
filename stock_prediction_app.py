import streamlit as st
from fetch_n_save_data import raw_stock_list, raw_stock_data
from data_processing import processed_stock_data
import datetime
import os
import glob
import pandas as pd


today = datetime.date.today()
today = today.strftime('%d-%m-%Y')

# pattern = os.path.join(download_dir, "stock_list_*.csv") ## this pattern is for stock list

# Date extraction from the existing files
download_dir = "downloaded_data"

# File pattern: raw_df_<symbol>_<date>.csv
pattern = os.path.join(download_dir, "raw_df_*_*.csv")
stock_files = glob.glob(pattern)

def extract_date_from_filename(filename):
    basename = os.path.basename(filename)
    try:
        parts = basename.replace(".csv", "").split("_")
        date_str = parts[-1]  # Last element should be date
        return datetime.datetime.strptime(date_str, "%d-%m-%Y")
    except Exception:
        return None  # Invalid format fallback

# Build list of (filename, date) pairs
dated_files = [(f, extract_date_from_filename(f)) for f in stock_files]
valid_files = [(f, d) for f, d in dated_files if d is not None]

# Sort and get the most recent one
if valid_files:
    latest_file, latest_date = sorted(valid_files, key=lambda x: x[1])[-1]
    latest_df = pd.read_csv(latest_file)
    last_refresh = latest_date.strftime("%d-%m-%Y")
else:
    latest_df = pd.DataFrame()
    last_refresh = "No valid file found"
# END of date extraction code

# Stock list Function call
raw_stock_list_df = raw_stock_list()

st.set_page_config(layout="wide")
st.header("My Stock Prediction App")
st.subheader('Stock List')
st.write(raw_stock_list_df)

def data_frm_streamlit():
    selected_stock = st.selectbox('Select a Stock Symbol from dropdown', raw_stock_list_df, placeholder="Select a Stock Symbol", index=None)
    selected_stock_data = raw_stock_list_df[raw_stock_list_df['symbol'] == selected_stock]
    selected_full_name = raw_stock_list_df.loc[raw_stock_list_df['symbol'] == selected_stock, 'name'].values
    if len(selected_full_name) > 0:
        selected_name = str(selected_full_name).strip("[]'\"")
    else:
        selected_name = "________"
    return selected_stock, selected_stock_data, selected_name

selected_stock,selected_stock_data, selected_full_name = data_frm_streamlit()


if selected_stock:

    st.write(selected_stock_data.reset_index(drop=True))
    st.subheader(f'Stock Data for {selected_full_name}')

    st.write(f"{selected_stock}'s last refresh date: {last_refresh} ,   Today's date: {today} " )

    # ✅ Initialize session state for refresh control
    if "refresh_stock_data" not in st.session_state:
        st.session_state.refresh_stock_data = None

    # 🧠 Display prompt
    st.text("Do you want to download fresh stock data?")

    # 🖱️ Create buttons with state-aware handlers
    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        if st.button("✅ Yes"):
            st.session_state.refresh_stock_data = "y"
    with col2:
        if st.button("❌ No"):
            st.session_state.refresh_stock_data = "n"

    # 🔄 Stock data logic runs ONLY after user choice
    if st.session_state.refresh_stock_data == "y":
        st.success("🔄 Fresh stock data will be downloaded.")
        raw_stock_df = raw_stock_data(user_input=selected_stock, refresh_stock_data="y")
        processed_stock_df = processed_stock_data(raw_stock_df)
        st.write(processed_stock_df)

    elif st.session_state.refresh_stock_data == "n":
        st.info("📁 Using local stock data.")
        raw_stock_df = raw_stock_data(user_input=selected_stock, refresh_stock_data="n")
        processed_stock_df = processed_stock_data(raw_stock_df)
        st.write(processed_stock_df)

    else:
        st.info("📌 Waiting for your choice: Download fresh data or use local.")
else:
    st.warning("⚠️ Please select a valid stock symbol from the dropdown.")



