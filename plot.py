import pandas as pd
import plotly.graph_objects as go

train_df = pd.read_csv("train.csv")

success_df = train_df[["episode_success", "step"]]
success_df = success_df.copy()
success_df["Success Rate"] = success_df["episode_success"].rolling(window=100).mean()

# Create Plotly figure
fig = go.Figure()

# Moving average line
fig.add_trace(go.Scatter(
    x=success_df['step'],
    y=success_df['Success Rate'],
    mode='lines',
    name='Success Rate'
))

fig.update_layout(
    title='Success Rate Over Time',
    xaxis_title='Step',
    yaxis_title='Success Rate',
    template='plotly_white'
)

fig.show()


