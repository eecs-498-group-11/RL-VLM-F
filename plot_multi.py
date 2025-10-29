import pandas as pd
import plotly.graph_objects as go


# folder called csvs with folder with this environment name, containing four csvs.
env_name = "sweep"
rolling = 100

# requires four train.csv files with data based on RLVLMF, VLM Score, GT Preference, and CLIP Score
rlvlmf_df = pd.read_csv("csvs/" + env_name + "/rlvlmf.csv")
rlvlmf_df = rlvlmf_df.iloc[:748][['episode_success', 'step']]
rlvlmf_df["Success Rate"] = rlvlmf_df["episode_success"].rolling(window=rolling).mean()

score_df = pd.read_csv("csvs/" + env_name + "/score.csv")
score_df = score_df[["episode_success", "step"]]
score_df["Success Rate"] = score_df["episode_success"].rolling(window=rolling).mean()

gt_df = pd.read_csv("csvs/" + env_name + "/gt.csv")
gt_df = gt_df.iloc[:748][['episode_success', 'step']]
gt_df["Success Rate"] = gt_df["episode_success"].rolling(window=rolling).mean()

clip_df = pd.read_csv("csvs/" + env_name + "/clip.csv")
clip_df = clip_df.iloc[:748][['episode_success', 'step']]
clip_df["Success Rate"] = clip_df["episode_success"].rolling(window=rolling).mean()

# Create Plotly figure
fig = go.Figure()

# RLVLMF line
fig.add_trace(go.Scatter(
    x=rlvlmf_df['step'],
    y=rlvlmf_df['Success Rate'],
    mode='lines',
    name='RLVLMF',
    line_width=2
))

# VLM Score line
fig.add_trace(go.Scatter(
    x=score_df['step'],
    y=score_df['Success Rate'],
    mode='lines',
    name='VLM Score',
    line_width=2
))

# GT line
fig.add_trace(go.Scatter(
    x=gt_df['step'],
    y=gt_df['Success Rate'],
    mode='lines',
    name='GT Preference',
    line_width=2
))

# CLIP line
fig.add_trace(go.Scatter(
    x=clip_df['step'],
    y=clip_df['Success Rate'],
    mode='lines',
    name='CLIP Score',
    line_width=2
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