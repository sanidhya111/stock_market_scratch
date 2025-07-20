import streamlit as st
from fetch_n_save_data import raw_stock_list, raw_stock_data
from data_processing import processed_stock_data, get_latest_stock_file_date, label_frm_columns, sanitized_data
import datetime
from data_visualization import stock_plot


today = datetime.date.today()
today = today.strftime('%d-%m-%Y')


# Stock list Function call
raw_stock_list_df = raw_stock_list()

st.set_page_config(layout="wide")
st.header("My Stock Prediction App")
st.subheader('Stock List')
st.write(raw_stock_list_df)


# Function to get the user stock selection from the Streamlit app
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


def handle_stock_selection(selected_stock, selected_stock_data, selected_full_name, today):
    st.write(selected_stock_data.reset_index(drop=True))
    st.subheader(f'Stock Data for {selected_full_name}')
    st.write(f"Today's date: {today}")

    _, last_refresh = get_latest_stock_file_date(selected_stock)
    st.write(f"{selected_stock}'s last refresh date: {last_refresh}")

    if "refresh_stock_data" not in st.session_state:
        st.session_state.refresh_stock_data = None

    if "prev_selected_stock" not in st.session_state:
        st.session_state.prev_selected_stock = None

    if selected_stock != st.session_state.prev_selected_stock:
        st.session_state.refresh_stock_data = None
        st.session_state.prev_selected_stock = selected_stock

    st.markdown("#### Do you want to download fresh stock data?")
    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        if st.button("✅ Yes"):
            st.session_state.refresh_stock_data = "y"
    with col2:
        if st.button("❌ No"):
            st.session_state.refresh_stock_data = "n"

    if st.session_state.refresh_stock_data == "y":
        st.success("🔄 Fresh stock data will be downloaded.")
        raw_stock_df = raw_stock_data(user_input=selected_stock, refresh_stock_data="y")
        processed_stock_df = processed_stock_data(raw_stock_df)
        _, column_names, dates, null_count_dict = label_frm_columns(processed_stock_df)
        datas, null_dict = sanitized_data(processed_stock_df)

        st.write(processed_stock_df)
        st.markdown(f"##### Showing the raw data null count below:")
        st.markdown(f"####### {null_count_dict}")
        st.markdown(f"##### Showing the total data below, after filling null with mean:")
        st.markdown(f"####### {null_dict}")

        fig = stock_plot(datas=datas, column_names=column_names, dates=dates)
        st.plotly_chart(fig, use_container_width=True)

        # After user selects "Yes"
        return selected_stock, st.session_state.refresh_stock_data

    elif st.session_state.refresh_stock_data == "n":
        st.info("📁 Using local stock data.")
        raw_stock_df = raw_stock_data(user_input=selected_stock, refresh_stock_data="n")
        processed_stock_df = processed_stock_data(raw_stock_df)
        _, column_names, dates, null_count_dict = label_frm_columns(processed_stock_df)
        datas, null_dict = sanitized_data(processed_stock_df)

        st.write(processed_stock_df)
        st.markdown(f"##### Showing the raw data null count below:")
        st.markdown(f"####### {null_count_dict}")
        st.markdown(f"##### Showing the total data below, after filling null with mean:")
        st.markdown(f"####### {null_dict}")

        fig = stock_plot(datas=datas, column_names=column_names, dates=dates)
        st.plotly_chart(fig, use_container_width=True)

        # After user selects "No"
        return selected_stock, st.session_state.refresh_stock_data

    else:
        st.info("📌 Waiting for your choice: Download fresh data or use local.")
        return None, None

if selected_stock:
    user_input, refresh_flag = handle_stock_selection(selected_stock, selected_stock_data, selected_full_name, today)
else:
    st.warning("⚠️ Please select a valid stock symbol from the dropdown.")