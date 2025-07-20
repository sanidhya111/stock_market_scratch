import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ENHANCED VERSION - maintaining your existing function signatures
def stock_plot(datas, column_names, dates=None, enhanced_styling=True):
    """
    ENHANCED: Your existing function with optional improvements
    - Maintains exact same interface as your original
    - Adds optional parameters for enhancements
    - Works with your existing main.py calls
    """
    rows = int(np.ceil(len(datas) / 2))
    fig, axs = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    axs = axs.flatten() if rows > 1 else [axs] if rows == 1 else axs

    for i, data in enumerate(datas):
        if dates is not None and len(dates) == len(data):
            x = dates
        else:
            x = np.linspace(0, len(data), len(data))

        axs[i].plot(x, data, linewidth=2 if enhanced_styling else 1)
        axs[i].set_title(column_names[i], pad=20)
        axs[i].set_xlabel("Date" if dates is not None else "Linear spaced")
        axs[i].set_ylabel(f"{column_names[i]} Value")
        axs[i].legend([column_names[i]])
        axs[i].grid(True, alpha=0.3 if enhanced_styling else True)

        # ENHANCEMENT: Add statistics if requested
        if enhanced_styling and len(data) > 0:
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            axs[i].text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                        transform=axs[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots if any
    for j in range(len(datas), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    return fig


def plot_regression_overlay(y_actual, y_predicted, title="Future Return",
                            xlabel="Sample Index", ylabel="Value",
                            show_metrics=True, confidence_intervals=None):
    """
    ENHANCED: Your existing function with additional optional features
    - Same core interface as your original
    - Added optional metrics display and confidence intervals
    """
    plt.figure(figsize=(12, 5))
    plt.plot(y_actual, label="Actual", color="blue", linewidth=2)
    plt.plot(y_predicted, label="Predicted", color="orange", linestyle="--", linewidth=2)

    # ENHANCEMENT: Optional confidence intervals
    if confidence_intervals is not None:
        plt.fill_between(range(len(y_predicted)),
                         y_predicted - confidence_intervals,
                         y_predicted + confidence_intervals,
                         alpha=0.2, color='orange', label='Confidence Interval')

    plt.title(f"{title}: Actual vs Predicted")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ENHANCEMENT: Optional metrics display
    if show_metrics:
        mse = np.mean((y_actual - y_predicted) ** 2)
        mae = np.mean(np.abs(y_actual - y_predicted))
        correlation = np.corrcoef(y_actual, y_predicted)[0, 1] if len(y_actual) > 1 else 0

        metrics_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nCorr: {correlation:.3f}'
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.show()


def plot_classification_overlay(y_actual, y_predicted, class_map=None,
                                title="Signal", xlabel="Sample Index", ylabel="Class",
                                show_confusion_matrix=True):
    """
    ENHANCED: Your existing function with optional confusion matrix
    - Maintains your exact interface
    - Adds optional confusion matrix visualization
    """
    # Handle mapping to labels if provided
    if class_map:
        actual_labels = [class_map[v] for v in y_actual]
        pred_labels = [class_map[v] for v in y_predicted]
    else:
        actual_labels = y_actual
        pred_labels = y_predicted

    if show_confusion_matrix:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))

    ax1.plot(actual_labels, label="Actual", marker='o', linestyle='-', color="green")
    ax1.plot(pred_labels, label="Predicted", marker='x', linestyle='--', color="red")
    ax1.set_title(f"{title}: Actual vs Predicted")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True)

    # ENHANCEMENT: Optional confusion matrix
    if show_confusion_matrix:
        cm = confusion_matrix(y_actual, y_predicted)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', ax=ax2,
                    xticklabels=list(class_map.values()) if class_map else None,
                    yticklabels=list(class_map.values()) if class_map else None)
        ax2.set_title("Confusion Matrix")
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")

    plt.tight_layout()
    plt.show()


def plot_histogram(data, bins=50, title="Distribution", xlabel="Value", ylabel="Frequency",
                   color="orange", edgecolor="black", clip_range=None, figsize=(10, 4),
                   show_stats=True):
    """
    ENHANCED: Your existing function with optional statistics
    - Same interface as your original
    - Added optional statistical information display
    """
    # Convert to Series and clean invalid values
    series = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna()

    # Clip outliers if requested
    if clip_range:
        series = series.clip(lower=clip_range[0], upper=clip_range[1])

    # Plot
    plt.figure(figsize=figsize)
    plt.hist(series, bins=bins, color=color, edgecolor=edgecolor, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    # ENHANCEMENT: Optional statistics
    if show_stats and len(series) > 0:
        stats_text = f'Count: {len(series)}\nMean: {series.mean():.4f}\nStd: {series.std():.4f}'
        plt.text(0.75, 0.95, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# Not used in main.py
def visualize_predictions(y_reg_test, y_pred_reg, y_pred_class, confidences,
                          enhanced_layout=True):
    """
    ENHANCED: Your existing function with better organization
    - Same interface as your original
    - Optional enhanced layout
    """
    plot_data_dict = {
        "Actual Future Return": y_reg_test,
        "Predicted Future Return": y_pred_reg,
        "Predicted Signal": y_pred_class,
        "Confidence Score": confidences
    }

    # Use your enhanced stock_plot function
    fig = stock_plot(
        datas=list(plot_data_dict.values()),
        column_names=list(plot_data_dict.keys()),
        enhanced_styling=enhanced_layout
    )
    plt.show()
    return fig


# NEW FUNCTION: Only adding genuinely new functionality
def plot_feature_importance(feature_names, importances, title="Feature Importance", top_n=15):
    """
    NEW: Additional function for feature importance (not replacing existing)
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)

    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title(f'{title} (Top {top_n})')
    plt.xlabel('Importance Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
