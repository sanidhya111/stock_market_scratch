import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def stock_plot(datas, column_names):
    """
    Plots multiple time series in subplots using matplotlib.

    Parameters:
    - datas: list of arrays or Series
    - column_names: list of strings (titles for each subplot)

    Returns:
    - fig: matplotlib Figure object
    """
    rows = int(np.ceil(len(datas) / 2))
    fig, axs = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    axs = axs.flatten()

    for i, data in enumerate(datas):
        x = np.linspace(0, len(data), len(data))
        axs[i].plot(x, data)
        axs[i].set_title(column_names[i], pad=20)
        axs[i].set_xlabel("Linear spaced")
        axs[i].set_ylabel(f"{column_names[i]} Value")
        axs[i].legend([column_names[i]])
        axs[i].grid(True)

    # Hide unused subplots if any
    for j in range(len(datas), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    return fig


def plot_regression_overlay(y_actual, y_predicted, title="Future Return", xlabel="Sample Index", ylabel="Value"):
    """
    Plots actual vs predicted regression output.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(y_actual, label="Actual", color="blue")
    plt.plot(y_predicted, label="Predicted", color="orange", linestyle="--")
    plt.title(f"{title}: Actual vs Predicted")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_classification_overlay(y_actual, y_predicted, class_map=None, title="Signal", xlabel="Sample Index", ylabel="Class"):
    """
    Plots actual vs predicted classification output using mapped labels.
    """
    # Handle mapping to labels if provided
    if class_map:
        actual_labels = [class_map[v] for v in y_actual]
        pred_labels = [class_map[v] for v in y_predicted]
    else:
        actual_labels = y_actual
        pred_labels = y_predicted

    plt.figure(figsize=(12, 4))
    plt.plot(actual_labels, label="Actual", marker='o', linestyle='-', color="green")
    plt.plot(pred_labels, label="Predicted", marker='x', linestyle='--', color="red")
    plt.title(f"{title}: Actual vs Predicted")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histogram(data, bins=50, title="Distribution", xlabel="Value", ylabel="Frequency",
                   color="orange", edgecolor="black", clip_range=None, figsize=(10, 4)):
    """
    Plots a histogram of the given numeric data.

    Parameters:
    - data: 1D array-like (e.g. list, NumPy array, or Series)
    - bins: number of histogram bins
    - title: plot title
    - xlabel, ylabel: axis labels
    - color, edgecolor: styling
    - clip_range: (min, max) tuple to clip extreme values
    - figsize: tuple for figure size
    """
    # Convert to Series and clean invalid values
    series = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna()

    # Clip outliers if requested
    if clip_range:
        series = series.clip(lower=clip_range[0], upper=clip_range[1])

    # Plot
    plt.figure(figsize=figsize)
    plt.hist(series, bins=bins, color=color, edgecolor=edgecolor)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_predictions(y_reg_test, y_pred_reg, y_pred_class, confidences):
    plot_data_dict = {
        "Actual Future Return": y_reg_test,
        "Predicted Future Return": y_pred_reg,
        "Predicted Signal": y_pred_class,
        "Confidence Score": confidences
    }
    fig = stock_plot(list(plot_data_dict.values()), list(plot_data_dict.keys()))
    plt.show()