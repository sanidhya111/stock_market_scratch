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