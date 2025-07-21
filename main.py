import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import datetime
from fetch_n_save_data import raw_stock_data
from data_processing import processed_stock_data, sanitized_data_df
from visualization_matplot import (stock_plot,
                                   plot_regression_overlay,
                                   plot_classification_overlay, visualize_predictions,
                                   plot_histogram)
import seaborn as sns
from ai_prediction import train_and_predict  # IMPORT FROM MODULE

# Your existing setup code
time_now = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
today = datetime.date.today().strftime("%d-%m-%Y")
print(f'The date and time now is: {time_now}')


# Use this code while using Streamlit only
from stock_prediction_app import data_frm_streamlit, handle_stock_selection
selected_stock,selected_stock_data, selected_full_name = data_frm_streamlit()
user_selection, refresh_flag = handle_stock_selection(selected_stock, selected_stock_data, selected_full_name, today)
raw_df = raw_stock_data(user_input=user_selection, refresh_stock_data=refresh_flag)


# Your existing data loading
# raw_df = raw_stock_data(user_input='A', refresh_stock_data='n')
processed_df = processed_stock_data(raw_df)
sanitized_df = sanitized_data_df(processed_df)

print(sanitized_df.head())

# AI PREDICTION - NOW JUST ONE FUNCTION CALL
results, message = train_and_predict(processed_df)

if results:
    prediction_df = results['prediction_df']
    test_data = results['test_data']
    metrics = results['metrics']
    signal_map = results['signal_map']

    print(f"\n📊 Classification Accuracy: {metrics['accuracy']:.4f}")
    print(f"📊 Regression R² Score: {metrics['r2']:.4f}")
    print(f"\n📊 Final Signal Distribution:")
    print(prediction_df["final_signal"].value_counts())

    # Your existing visualization calls
    plot_data_dict = {
        "Actual Future Return": test_data['y_reg_test'].values,
        "Predicted Future Return": test_data['y_pred_reg'],
        "Predicted Signal": test_data['y_pred_class'],
        "Confidence Score": test_data['confidences']
    }

    fig = stock_plot(
        datas=list(plot_data_dict.values()),
        column_names=list(plot_data_dict.keys())
    )
    # plt.show()

    plot_regression_overlay(
        y_actual=test_data['y_reg_test'].values,
        y_predicted=test_data['y_pred_reg'],
        title="Future Return",
        xlabel="Test Sample Index",
        ylabel="Return"
    )

    plot_classification_overlay(
        y_actual=test_data['y_class_test'],
        y_predicted=test_data['y_pred_class'],
        class_map=signal_map,
        title="Signal",
        xlabel="Test Sample Index",
        ylabel="Signal"
    )

    visualize_predictions(
        test_data['y_reg_test'].values,
        test_data['y_pred_reg'],
        test_data['y_pred_class'],
        test_data['confidences']
    )

else:
    print(f"AI Prediction failed: {message}")

