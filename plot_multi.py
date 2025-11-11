import pandas as pd
import plotly.graph_objects as go
import numpy as np

# requires four train.csv files with data based on RLVLMF, VLM Score, GT Preference, and CLIP Score
folder = "data/plotting_data/"
rlvlmf_df = pd.read_csv(folder+"rlvlmf.csv")
rlvlmf_df = rlvlmf_df[["episode_success", "step"]]
rlvlmf_df["Success Rate"] = rlvlmf_df["episode_success"].rolling(window=100).mean()
rlvlmf_se = np.sqrt(rlvlmf_df["Success Rate"] * (1 - rlvlmf_df["Success Rate"]) / 100)

score_df = pd.read_csv(folder+"score.csv")
score_df = score_df[["episode_success", "step"]]
score_df["Success Rate"] = score_df["episode_success"].rolling(window=100).mean()
score_se = np.sqrt(score_df["Success Rate"] * (1 - score_df["Success Rate"]) / 100)

gt_df = pd.read_csv(folder+"gt.csv")
gt_df = gt_df[["episode_success", "step"]]
gt_df["Success Rate"] = gt_df["episode_success"].rolling(window=100).mean()
gt_se = np.sqrt(gt_df["Success Rate"] * (1 - gt_df["Success Rate"]) / 100)

clip_df = pd.read_csv(folder+"clip.csv")
clip_df = clip_df[["episode_success", "step"]]
clip_df["Success Rate"] = clip_df["episode_success"].rolling(window=100).mean()
clip_se = np.sqrt(clip_df["Success Rate"] * (1 - clip_df["Success Rate"]) / 100)

a_df = pd.read_csv(folder+"extension_a.csv")
a_df = a_df[["episode_success", "step"]]
a_df["Success Rate"] = a_df["episode_success"].rolling(window=100).mean()
a_se = np.sqrt(a_df["Success Rate"] * (1 - a_df["Success Rate"]) / 100)

single_df = pd.read_csv(folder+"single_prompt.csv")
single_df = single_df[["episode_success", "step"]]
single_df["Success Rate"] = single_df["episode_success"].rolling(window=100).mean()
single_se = np.sqrt(single_df["Success Rate"] * (1 - single_df["Success Rate"]) / 100)

# Create Plotly figure
fig = go.Figure([
go.Scatter(
        x=rlvlmf_df['step'],
        y=rlvlmf_df['Success Rate']- rlvlmf_se,
        line=dict(color='red', width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=rlvlmf_df['step'],
        y=rlvlmf_df['Success Rate'] + rlvlmf_se,
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='red', width=0),
        name='Standard Error'
    ),
go.Scatter(
        x=score_df['step'],
        y=score_df['Success Rate']- score_se,
        line=dict(color='purple', width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=score_df['step'],
        y=score_df['Success Rate'] + score_se,
        fill='tonexty',
        fillcolor='rgba(100,0,100,0.2)',
        line=dict(color='purple', width=0),
        name='Standard Error'
    ),
go.Scatter(
        x=gt_df['step'],
        y=gt_df['Success Rate']- gt_se,
        line=dict(color='green', width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=gt_df['step'],
        y=gt_df['Success Rate'] + gt_se,
        fill='tonexty',
        fillcolor='rgba(0,255,0,0.2)',
        line=dict(color='green', width=0),
        name='Standard Error'
    ),
go.Scatter(
        x=clip_df['step'],
        y=clip_df['Success Rate']- clip_se,
        line=dict(color='blue', width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=clip_df['step'],
        y=clip_df['Success Rate'] + clip_se,
        fill='tonexty',
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(color='blue', width=0),
        name='Standard Error'
    ),
go.Scatter(
        x=a_df['step'],
        y=a_df['Success Rate']- a_se,
        line=dict(color='orange', width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=a_df['step'],
        y=a_df['Success Rate'] + a_se,
        fill='tonexty',
        fillcolor='rgba(255,128,0,0.2)',
        line=dict(color='orange', width=0),
        name='Standard Error'
    ),
go.Scatter(
        x=single_df['step'],
        y=single_df['Success Rate']- single_se,
        line=dict(color='teal', width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=single_df['step'],
        y=single_df['Success Rate'] + single_se,
        fill='tonexty',
        fillcolor='rgba(0,153,153,0.2)',
        line=dict(color='teal', width=0),
        name='Standard Error'
    )])

# RLVLMF line
fig.add_trace(go.Scatter(
    x=rlvlmf_df['step'],
    y=rlvlmf_df['Success Rate'],
    mode='lines',
    line=dict(color='red', width=2),
    name='RLVLMF'
))

# VLM Score line
fig.add_trace(go.Scatter(
    x=score_df['step'],
    y=score_df['Success Rate'],
    mode='lines',
    line=dict(color='purple', width=2),
    name='VLM Score'
))

# GT line
fig.add_trace(go.Scatter(
    x=gt_df['step'],
    y=gt_df['Success Rate'],
    mode='lines',
    line=dict(color='green', width=2),
    name='GT Preference'
))

# CLIP line
fig.add_trace(go.Scatter(
    x=clip_df['step'],
    y=clip_df['Success Rate'],
    mode='lines',
    line=dict(color='blue', width=2),
    name='CLIP Score'
))

# Extension A line
fig.add_trace(go.Scatter(
    x=a_df['step'],
    y=a_df['Success Rate'],
    mode='lines',
    line=dict(color='orange', width=2),
    name='Extension A Free Form Score'
))

# Extension A Single Prompt line
fig.add_trace(go.Scatter(
    x=single_df['step'],
    y=single_df['Success Rate'],
    mode='lines',
    line=dict(color='teal', width=2),
    name='Extension A Single Prompt Score'
))

fig.update_layout(
    title='Success Rate Over Time',
    xaxis=dict(
        title='Step',
        dtick=50000,
        range=[0,300000]
    ),
    yaxis=dict(
        title='Success Rate',
        dtick=.1,
        range=[0,1]
    ),
    template='plotly_white'
)

# show and/or save figure
fig.show()
#fig.write_image("drawer_fig4.png")
