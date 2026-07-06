import json
import os

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "all_runs.json"
)


def load_runs():
    if not os.path.exists(RESULTS_PATH):
        return pd.DataFrame()
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return pd.DataFrame(data)


app = dash.Dash(__name__)
app.title = "ML-Inference-Bench Dashboard"

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "maxWidth": "1000px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H1("ML-Inference-Bench"),
        html.P("Comparing latency, throughput, memory, and accuracy across models and precisions."),
        dcc.Interval(id="refresh", interval=5000, n_intervals=0),
        html.Div(id="content"),
    ],
)


@app.callback(Output("content", "children"), Input("refresh", "n_intervals"))
def update(_):
    df = load_runs()
    if df.empty:
        return html.P(
            "No runs found yet. Run `python main.py --model resnet50 --precision fp16` "
            "(or any model/precision combo) at least once to populate this dashboard."
        )

    df["label"] = df["model"] + " / " + df["precision"]

    latency_fig = go.Figure()
    latency_fig.add_bar(x=df["label"], y=df["p50_ms"], name="p50 (ms)")
    latency_fig.add_bar(x=df["label"], y=df["p95_ms"], name="p95 (ms)")
    latency_fig.add_bar(x=df["label"], y=df["p99_ms"], name="p99 (ms)")
    latency_fig.update_layout(title="Latency by model / precision", barmode="group")

    throughput_fig = go.Figure()
    for col in ["batch_1", "batch_8", "batch_32"]:
        if col in df.columns:
            throughput_fig.add_bar(x=df["label"], y=df[col], name=col)
    throughput_fig.update_layout(title="Throughput (images/sec) by batch size", barmode="group")

    memory_fig = go.Figure()
    memory_fig.add_bar(x=df["label"], y=df["memory_mb"])
    memory_fig.update_layout(title="Peak GPU memory (MB)")

    accuracy_fig = go.Figure()
    accuracy_fig.add_bar(x=df["label"], y=df["cosine_similarity"])
    accuracy_fig.update_layout(
        title="Accuracy retained vs. FP32 baseline (cosine similarity)", yaxis_range=[0, 1]
    )

    return html.Div(
        [
            dcc.Graph(figure=latency_fig),
            dcc.Graph(figure=throughput_fig),
            dcc.Graph(figure=memory_fig),
            dcc.Graph(figure=accuracy_fig),
        ]
    )


def run():
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    run()