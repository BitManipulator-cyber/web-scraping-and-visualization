import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np

# ===== Dummy dataset =====
years = pd.date_range("1990", periods=36, freq="YE")
data = {
    "date": years,
    "actual_revenue": np.linspace(50, 200, 36) + np.random.randn(36) * 10,
    "predicted_revenue": np.linspace(55, 210, 36) + np.random.randn(36) * 8,
    "passenger_return_rate": np.random.uniform(60, 90, 36),
}
df = pd.DataFrame(data)

# ===== Create a Dash app =====
app = dash.Dash(__name__)

# ===== Layout =====
app.layout = html.Div([
    html.H1("Sample Flight Dashboard", style={"textAlign": "center"}),

    # Dropdown for selecting metric
    html.Label("Select Metric:", style={"fontWeight": "bold"}),
    dcc.Dropdown(
        id="metric-dropdown",
        options=[
            {"label": "Revenue (Actual vs Predicted)", "value": "revenue"},
            {"label": "Passenger Return Rate", "value": "return_rate"}
        ],
        value="revenue"
    ),

    # Graph output
    dcc.Graph(id="main-graph")
])

# ===== Callback for interactivity =====
@app.callback(
    dash.Output("main-graph", "figure"),
    [dash.Input("metric-dropdown", "value")]
)
def update_graph(selected_metric):
    if selected_metric == "revenue":
        fig = px.line(df, x="date", y=["actual_revenue", "predicted_revenue"],
                      title="Actual vs Predicted Revenue")
    else:
        fig = px.line(df, x="date", y="passenger_return_rate",
                      title="Passenger Return Rate Over Time")

    fig.update_layout(xaxis_title="Year", yaxis_title="Value")
    return fig

# ===== Run server =====
if __name__ == "__main__":
    app.run(debug=True)
