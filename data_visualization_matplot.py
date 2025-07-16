# import matplotlib.pyplot as plt
# import numpy as np
#
# def stock_plot(datas, column_names):
#     rows = int(np.ceil(len(datas) / 2))
#     fig, axs = plt.subplots(rows, 2, figsize=(12, 4 * rows))
#     axs = axs.flatten()
#
#     for i, data in enumerate(datas):
#         x = np.linspace(0, len(data), len(data))
#         axs[i].plot(x, data)  # Label removed because wanted to use column name directly
#     for j, name in enumerate(column_names):
#         axs[j].set_title(name, pad=20)
#         axs[j].set_xlabel("Linear spaced")
#         axs[j].set_ylabel(f"{name} Price")
#         axs[j].legend()
#         axs[j].grid(True)
#     plt.tight_layout()
#     return fig


# As matplotlib is not interactive on STREAMLIT, changing over to Plotly

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def stock_plot(datas, column_names):
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
        x = list(range(len(data)))  # Linear spacing since timestamps aren't used yet
        label = column_names[i].replace("📈 ", "").replace(" Price", "")

        fig.add_trace(
            go.Scatter(x=x, y=data, mode="lines", name=label, showlegend=True),
            row=row, col=col
        )

        # Add axis labels for each subplot
        fig.update_xaxes(title_text="Linear spaced", row=row, col=col)
        fig.update_yaxes(title_text=f"{label} Price", row=row, col=col)

    fig.update_layout(
        height=rows * 350,
        title_text="📊 Interactive Stock Metrics",
        hovermode="x unified",
        legend_title_text="Metric",
        template="plotly_white"
    )

    return fig