import pandas as pd
import plotly.graph_objects as go
import numpy as np

# folder called csvs with folder with this environment name, containing four csvs.
env_name = "soccer_b"
rolling = 100

# requires six train.csv files with data based on RLVLMF (w/ and w/o metadata), VLM Score (w/ and w/o metadata), GT Preference, and CLIP Score
rlvlmf_w_df = pd.read_csv("csvs/" + env_name + "/rlvlmf_w.csv")
rlvlmf_w__df = rlvlmf_w_df[rlvlmf_w_df["TimeLimit.truncated"] != 0][['episode_success', 'step']]
rlvlmf_w_df["Success Rate"] = rlvlmf_w_df["episode_success"].rolling(window=rolling).mean()
rlvlmf_w_se = np.sqrt(rlvlmf_w_df["Success Rate"] * (1 - rlvlmf_w_df["Success Rate"]) / 100)

rlvlmf_wo_df = pd.read_csv("csvs/" + env_name + "/rlvlmf_wo.csv")
rlvlmf_wo_df = rlvlmf_wo_df[rlvlmf_wo_df["TimeLimit.truncated"] != 0][['episode_success', 'step']]
rlvlmf_wo_df["Success Rate"] = rlvlmf_wo_df["episode_success"].rolling(window=rolling).mean()
rlvlmf_wo_se = np.sqrt(rlvlmf_wo_df["Success Rate"] * (1 - rlvlmf_wo_df["Success Rate"]) / 100)

score_w_df = pd.read_csv("csvs/" + env_name + "/score_w.csv")
score_w_df = score_w_df[score_w_df["TimeLimit.truncated"] != 0][["episode_success", "step"]]
score_w_df["Success Rate"] = score_w_df["episode_success"].rolling(window=rolling).mean()
score_w_se = np.sqrt(score_w_df["Success Rate"] * (1 - score_w_df["Success Rate"]) / 100)

score_wo_df = pd.read_csv("csvs/" + env_name + "/score_wo.csv")
score_wo_df = score_wo_df[score_wo_df["TimeLimit.truncated"] != 0][["episode_success", "step"]]
score_wo_df["Success Rate"] = score_wo_df["episode_success"].rolling(window=rolling).mean()
score_wo_se = np.sqrt(score_wo_df["Success Rate"] * (1 - score_wo_df["Success Rate"]) / 100)

gt_df = pd.read_csv("csvs/" + env_name + "/gt.csv")
gt_df = gt_df[gt_df["TimeLimit.truncated"] != 0][['episode_success', 'step']]
gt_df["Success Rate"] = gt_df["episode_success"].rolling(window=rolling).mean()
gt_se = np.sqrt(gt_df["Success Rate"] * (1 - gt_df["Success Rate"]) / 100)

clip_df = pd.read_csv("csvs/" + env_name + "/clip.csv")
clip_df = clip_df[clip_df["TimeLimit.truncated"] != 0][['episode_success', 'step']]
clip_df["Success Rate"] = clip_df["episode_success"].rolling(window=rolling).mean()
clip_se = np.sqrt(clip_df["Success Rate"] * (1 - clip_df["Success Rate"]) / 100)

# Create Plotly figure
fig = go.Figure([
go.Scatter(
        x=rlvlmf_w_df['step'],
        y=rlvlmf_w_df['Success Rate']- rlvlmf_w_se,
        line=dict(color='orange', width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=rlvlmf_w_df['step'],
        y=rlvlmf_w_df['Success Rate']+ rlvlmf_w_se,
        line=dict(color='orange', width=0),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.2)',
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=rlvlmf_wo_df['step'],
        y=rlvlmf_wo_df['Success Rate'] - rlvlmf_wo_se,
        showlegend=False,
        hoverinfo='skip',
        line=dict(color='red', width=0)
    ),
go.Scatter(
        x=rlvlmf_wo_df['step'],
        y=rlvlmf_wo_df['Success Rate'] + rlvlmf_wo_se,
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='red', width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=score_w_df['step'],
        y=score_w_df['Success Rate']- score_w_se,
        line=dict(color='magenta', width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=score_w_df['step'],
        y=score_w_df['Success Rate'] + score_w_se,
        line=dict(color='magenta', width=0),
        fill='tonexty',
        fillcolor='rgba(255,0,255,0.2)',
        showlegend=False,
        hoverinfo='skip'
    ),
go.Scatter(
        x=score_wo_df['step'],
        y=score_wo_df['Success Rate'] - score_wo_se,
        showlegend=False,
        hoverinfo='skip',
        line=dict(color='purple', width=0),
    ),
go.Scatter(
        x=score_wo_df['step'],
        y=score_wo_df['Success Rate'] + score_wo_se,
        fill='tonexty',
        fillcolor='rgba(100,0,100,0.2)',
        showlegend=False,
        hoverinfo='skip',
        line=dict(color='purple', width=0),
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
    )])

# RLVLMF line
fig.add_trace(go.Scatter(
    x=rlvlmf_w_df['step'],
    y=rlvlmf_w_df['Success Rate'],
    mode='lines',
    line=dict(color='orange', width=2),
    name='RLVLMF w/ metadata'
))

# RLVLMF line
fig.add_trace(go.Scatter(
    x=rlvlmf_wo_df['step'],
    y=rlvlmf_wo_df['Success Rate'],
    mode='lines',
    line=dict(color='red', width=2),
    name='RLVLMF w/o metadata'
))

# VLM Score line
fig.add_trace(go.Scatter(
    x=score_w_df['step'],
    y=score_w_df['Success Rate'],
    mode='lines',
    line=dict(color='magenta', width=2),
    name='VLM Score w/ metadata'
))

# VLM Score line
fig.add_trace(go.Scatter(
    x=score_wo_df['step'],
    y=score_wo_df['Success Rate'],
    mode='lines',
    line=dict(color='purple', width=2),
    name='VLM Score w/o metadata'
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

fig.update_layout(
    title='Success Rate Over Time: {}'.format(env_name),
    xaxis=dict(
        title='Step',
        dtick=50000,
        range=[50000,250000]
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