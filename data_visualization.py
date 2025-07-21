# As matplotlib is not interactive on STREAMLIT, changing over to Plotly

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def stock_plot(datas, column_names, dates):
    rows = int((len(datas) + 1) // 2)
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=column_names,
        shared_xaxes=False,
        shared_yaxes=False
    )

    for i, data in enumerate(datas):
        row = (i // 2) + 1
        col = (i % 2) + 1

        # Use real dates if available, fallback to linear spacing
        x = dates if dates is not None else list(range(len(data)))

        label = column_names[i].replace("📈 ", "").replace(" Price", "")

        fig.add_trace(
            go.Scatter(
                x=x,
                y=data,
                mode="lines",
                name=label,
                showlegend=True,
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}" if dates is not None else "%{x}<br>%{y:.2f}"
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(title_text="Date" if dates is not None else "Linear spaced", row=row, col=col)
        fig.update_yaxes(title_text=f"{label} Price", row=row, col=col)

    fig.update_layout(
        height=rows * 350,
        title_text="📊 Interactive Stock Metrics",
        hovermode="x unified",
        legend_title_text="Metric",
        template="plotly_white"
    )

    return fig



def plot_regression_overlay_plotly(y_actual, y_predicted, title="Future Return",
                                   xlabel="Sample Index", ylabel="Value"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_actual, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=y_predicted, mode='lines', name='Predicted', line=dict(color='orange', dash='dash')))
    fig.update_layout(
        title=f"{title}: Actual vs Predicted",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white"
    )
    return fig

def plot_classification_overlay_plotly(y_actual, y_predicted, class_map=None,
                                       title="Signal", xlabel="Sample Index", ylabel="Class"):
    def map_labels(arr):
        return [class_map[v] if class_map else str(v) for v in arr]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=map_labels(y_actual), mode='lines+markers', name='Actual', line=dict(color='green')))
    fig.add_trace(go.Scatter(y=map_labels(y_predicted), mode='lines+markers', name='Predicted', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title=f"{title}: Actual vs Predicted",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white"
    )
    return fig

def visualize_predictions_plotly(y_reg_test, y_pred_reg, y_pred_class, confidences, class_map=None):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, subplot_titles=[
        "Actual Future Return", "Predicted Future Return", "Predicted Signal", "Confidence Score"])
    fig.add_trace(go.Scatter(y=y_reg_test, mode='lines', name="Actual Future Return", line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=y_pred_reg, mode='lines', name="Predicted Future Return", line=dict(color='orange')), row=2, col=1)
    pred_labels = [class_map[c] if class_map else str(c) for c in y_pred_class]
    fig.add_trace(go.Bar(y=pred_labels, name='Predicted Signal', marker_color='green'), row=3, col=1)
    fig.add_trace(go.Scatter(y=confidences, mode='lines', name="Confidence Score", line=dict(color='purple')), row=4, col=1)
    fig.update_layout(height=900, template="plotly_white")
    return fig
