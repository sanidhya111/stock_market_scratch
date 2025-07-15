import streamlit as st
from fetch_n_save_data import raw_stock_list
from data_processing import processed_stock_data

# Function calls
raw_stock_list_df = raw_stock_list()
stock_processed_data = processed_stock_data(raw_stock_list_df)


st.set_page_config(layout="wide")
st.header("My Stock Prediction App")
st.subheader('Stock List')
st.write(raw_stock_list_df)

def data_frm_streamlit(raw_stock_list_df):
    selected_stock = st.selectbox('Select a Stock Symbol from dropdown', raw_stock_list_df, placeholder="Select a Stock Symbol", index=None)
    selected_stock_data = raw_stock_list_df[raw_stock_list_df['symbol'] == selected_stock]
    selected_name = raw_stock_list_df.loc[raw_stock_list_df['symbol'] == selected_stock, 'name'].values
    if len(selected_name) > 0:
        selected_name = str(selected_name).strip("[]'\"")
    else:
        selected_name = "________"
    return selected_stock,selected_stock_data, selected_name

selected_stock, selected_stock_data, selected_name = data_frm_streamlit(raw_stock_list_df)

st.write(selected_stock_data.reset_index(drop=True))
st.subheader(f'Stock Data for {selected_name}')
st.write(stock_processed_data)