# stock_prediction_app.py - YOUR EXISTING CODE + MODULAR AI IMPORT
import streamlit as st
from fetch_n_save_data import raw_stock_list, raw_stock_data
from data_processing import processed_stock_data, get_latest_stock_file_date, label_frm_columns, sanitized_data
import datetime
from data_visualization import stock_plot
from ai_prediction import train_and_predict  # IMPORT FROM MODULE
from visualization_matplot import (plot_regression_overlay, plot_classification_overlay,
                                   plot_histogram, visualize_predictions)
import matplotlib.pyplot as plt

# ALL YOUR EXISTING CODE REMAINS EXACTLY THE SAME
today = datetime.date.today()
today = today.strftime('%d-%m-%Y')

raw_stock_list_df = raw_stock_list()

st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.header("My Stock Prediction App")

with st.expander("📋 Available Stock List", expanded=True):
    st.dataframe(raw_stock_list_df, use_container_width=True)

def data_frm_streamlit():
    selected_stock = st.selectbox('Select a Stock Symbol from dropdown', raw_stock_list_df,
                                  placeholder="Select a Stock Symbol", index=None)
    selected_stock_data = raw_stock_list_df[raw_stock_list_df['symbol'] == selected_stock]
    selected_full_name = raw_stock_list_df.loc[raw_stock_list_df['symbol'] == selected_stock, 'name'].values

    if len(selected_full_name) > 0:
        selected_name = str(selected_full_name).strip("[]'\"")
    else:
        selected_name = "________"
    return selected_stock, selected_stock_data, selected_name


selected_stock, selected_stock_data, selected_full_name = data_frm_streamlit()


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

        return selected_stock, st.session_state.refresh_stock_data, processed_stock_df

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

        return selected_stock, st.session_state.refresh_stock_data, processed_stock_df

    else:
        st.info("📌 Waiting for your choice: Download fresh data or use local.")
        return None, None, None


# YOUR EXISTING EXECUTION
if selected_stock:
    user_input, refresh_flag, processed_stock_df = handle_stock_selection(selected_stock, selected_stock_data,
                                                                          selected_full_name, today)
else:
    st.warning("⚠️ Please select a valid stock symbol from the dropdown.")
    processed_stock_df = None

# AI SECTION - NOW PROPERLY MODULAR (JUST ONE FUNCTION CALL)
if selected_stock and processed_stock_df is not None:
    st.markdown("---")
    st.header("🤖 AI Prediction Analysis")

    if st.button("🚀 Run AI Analysis", type="primary"):
        with st.spinner("Running AI prediction analysis..."):
            # SINGLE FUNCTION CALL - NO CODE DUPLICATION
            results, message = train_and_predict(processed_stock_df)

            if results:
                st.success("✅ AI Analysis completed!")

                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{results['metrics']['accuracy']:.1%}")
                with col2:
                    st.metric("R² Score", f"{results['metrics']['r2']:.4f}")
                with col3:
                    latest_signal = results['prediction_df']['final_signal'].iloc[-1]
                    st.metric("Latest Signal", latest_signal)

                # Show predictions
                st.subheader("Recent Predictions")
                st.dataframe(results['prediction_df'].tail(10))

                # Use YOUR visualization functions
                test_data = results['test_data']

                with st.expander("Regression Analysis"):
                    fig1 = plt.figure(figsize=(12, 5))
                    plot_regression_overlay(
                        y_actual=test_data['y_reg_test'].values,
                        y_predicted=test_data['y_pred_reg'],
                        title="Future Return",
                        xlabel="Test Sample Index",
                        ylabel="Return"
                    )
                    st.pyplot(fig1)
                    plt.close()

                with st.expander("Classification Analysis"):
                    fig2 = plt.figure(figsize=(12, 4))
                    plot_classification_overlay(
                        y_actual=test_data['y_class_test'],
                        y_predicted=test_data['y_pred_class'],
                        class_map=results['signal_map'],
                        title="Signal",
                        xlabel="Test Sample Index",
                        ylabel="Signal"
                    )
                    st.pyplot(fig2)
                    plt.close()

                with st.expander("Complete Overview"):
                    fig3 = plt.figure(figsize=(15, 12))
                    visualize_predictions(
                        test_data['y_reg_test'].values,
                        test_data['y_pred_reg'],
                        test_data['y_pred_class'],
                        test_data['confidences']
                    )
                    st.pyplot(fig3)
                    plt.close()
            else:
                st.error(f"❌ Analysis failed: {message}")
