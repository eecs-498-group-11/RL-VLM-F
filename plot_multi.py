import pandas as pd
import plotly.graph_objects as go

# requires four train.csv files with data based on RLVLMF, VLM Score, GT Preference, and CLIP Score
rlvlmf_df = pd.read_csv("rlvlmf.csv")
rlvlmf_df = rlvlmf_df[["episode_success", "step"]]
rlvlmf_df["Success Rate"] = rlvlmf_df["episode_success"].rolling(window=100).mean()

score_df = pd.read_csv("score.csv")
score_df = score_df[["episode_success", "step"]]
score_df["Success Rate"] = score_df["episode_success"].rolling(window=100).mean()

gt_df = pd.read_csv("gt.csv")
gt_df = gt_df[["episode_success", "step"]]
gt_df["Success Rate"] = gt_df["episode_success"].rolling(window=100).mean()

clip_df = pd.read_csv("clip.csv")
clip_df = clip_df[["episode_success", "step"]]
clip_df["Success Rate"] = clip_df["episode_success"].rolling(window=100).mean()

# Create Plotly figure
fig = go.Figure()

# RLVLMF line
fig.add_trace(go.Scatter(
    x=rlvlmf_df['step'],
    y=rlvlmf_df['Success Rate'],
    mode='lines',
    name='RLVLMF'
))

# VLM Score line
fig.add_trace(go.Scatter(
    x=score_df['step'],
    y=score_df['Success Rate'],
    mode='lines',
    name='VLM Score'
))

# GT line
fig.add_trace(go.Scatter(
    x=gt_df['step'],
    y=gt_df['Success Rate'],
    mode='lines',
    name='GT Preference'
))

# CLIP line
fig.add_trace(go.Scatter(
    x=clip_df['step'],
    y=clip_df['Success Rate'],
    mode='lines',
    name='CLIP Score'
))

fig.update_layout(
    title='Success Rate Over Time',
    xaxis_title='Step',
    yaxis_title='Success Rate',
    template='plotly_white'
)

# show and/or save figure
fig.show()
#fig.write_image("drawer_fig4.png")