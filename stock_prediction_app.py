import streamlit as st
from fetch_n_save_data import raw_stock_list, raw_stock_data
from data_processing import processed_stock_data
import datetime
import os
import glob
import re
from data_visualization_matplot import stock_plot


today = datetime.date.today()
today = today.strftime('%d-%m-%Y')

# pattern = os.path.join(download_dir, "stock_list_*.csv") ## this pattern is for stock list

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
    st.write(f"Today's date: {today}")

    # 🧠 Get refresh date only for selected symbol
    _, last_refresh = get_latest_stock_file_date(selected_stock)
    st.write(f"{selected_stock}'s last refresh date: {last_refresh}")


    # ✅ Initialize session keys
    if "refresh_stock_data" not in st.session_state:
        st.session_state.refresh_stock_data = None

    if "prev_selected_stock" not in st.session_state:
        st.session_state.prev_selected_stock = None

    # 🔁 Reset refresh choice on new selection
    if selected_stock and selected_stock != st.session_state.prev_selected_stock:
        st.session_state.refresh_stock_data = None
        st.session_state.prev_selected_stock = selected_stock

    # 🧠 Display prompt
    st.markdown("#### Do you want to download fresh stock data?") # Text smaller than subheader

    # 🖱️ Buttons update session state only on click
    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        if st.button("✅ Yes"):
            st.session_state.refresh_stock_data = "y"
    with col2:
        if st.button("❌ No"):
            st.session_state.refresh_stock_data = "n"

    # 🔄 Run logic only after user makes a decision
    if st.session_state.refresh_stock_data == "y":
        st.success("🔄 Fresh stock data will be downloaded.")
        raw_stock_df = raw_stock_data(
            user_input=selected_stock,
            refresh_stock_data="y"
        )
        processed_stock_df = processed_stock_data(raw_stock_df)
        st.write(processed_stock_df)


        # Extract target columns from DataFrame
        stock_columns_list = processed_stock_df.columns[:-1]

        # Prepare data and labels
        datas = [processed_stock_df[col] for col in stock_columns_list]
        column_names = [f"📈 {col.split('. ')[-1].capitalize()} Price" for col in stock_columns_list]

        # Call the Plotly version
        fig = stock_plot(datas=datas, column_names=column_names)
        st.plotly_chart(fig, use_container_width=True)



    elif st.session_state.refresh_stock_data == "n":
        st.info("📁 Using local stock data.")
        raw_stock_df = raw_stock_data(
            user_input=selected_stock,
            refresh_stock_data="n"
        )
        processed_stock_df = processed_stock_data(raw_stock_df)
        st.write(processed_stock_df)

        # Extract target columns from DataFrame
        stock_columns_list = processed_stock_df.columns[:-1]

        # Prepare data and labels
        datas = [processed_stock_df[col] for col in stock_columns_list]
        column_names = [f"📈 {col.split('. ')[-1].capitalize()} Price" for col in stock_columns_list]

        # Call the Plotly version
        fig = stock_plot(datas=datas, column_names=column_names)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📌 Waiting for your choice: Download fresh data or use local.")
else:
    st.warning("⚠️ Please select a valid stock symbol from the dropdown.")



