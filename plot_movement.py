import pandas as pd
import plotly.graph_objects as go
import numpy as np

def smooth_and_downsample(df, window=100, rate=50):
    df["total_movement"] = df["total_movement"].rolling(window).mean()
    # Return the downsampled version ONLY for plotting
    return df.iloc[::rate]

# requires four train.csv files with data based on RLVLMF, VLM Score, GT Preference, and CLIP Score
folder = "csvs/soccer_b/"
rlvlmf_df = pd.read_csv(folder+"rlvlmf_w.csv")
rlvlmf_df = rlvlmf_df[["total_movement", "step"]]
#rlvlmf_df = smooth_and_downsample(rlvlmf_df)
rlvlmf_df["total_movement"] = rlvlmf_df["total_movement"].rolling(window=100).mean()
#rlvlmf_se = np.sqrt(rlvlmf_df["total_movement"] * (1 - rlvlmf_df["total_movement"]) / 100)

rlvlmf_wo_df = pd.read_csv(folder+"rlvlmf_wo.csv")
rlvlmf_wo_df = rlvlmf_wo_df[["total_movement", "step"]]
#rlvlmf_wo_df = smooth_and_downsample(rlvlmf_wo_df)
rlvlmf_wo_df["total_movement"] = rlvlmf_wo_df["total_movement"].rolling(window=100).mean()
#rlvlmf_wo_se = np.sqrt(rlvlmf_wo_df["total_movement"] * (1 - rlvlmf_wo_df["total_movement"]) / 100)

score_df = pd.read_csv(folder+"score_w.csv")
score_df = score_df[["total_movement", "step"]]
#score_df = smooth_and_downsample(score_df)
score_df["total_movement"] = score_df["total_movement"].rolling(window=100).mean()
#score_se = np.sqrt(score_df["total_movement"] * (1 - score_df["total_movement"]) / 100)

score_wo_df = pd.read_csv(folder+"score_wo.csv")
score_wo_df = score_wo_df[["total_movement", "step"]]
#score_wo_df = smooth_and_downsample(score_wo_df)
score_wo_df["total_movement"] = score_wo_df["total_movement"].rolling(window=100).mean()
#score_wo_se = np.sqrt(score_wo_df["total_movement"] * (1 - score_wo_df["total_movement"]) / 100)

gt_df = pd.read_csv(folder+"gt.csv")
gt_df = gt_df[["total_movement", "step"]]
#gt_df = smooth_and_downsample(gt_df)
gt_df["total_movement"] = gt_df["total_movement"].rolling(window=100).mean()
#gt_se = np.sqrt(gt_df["total_movement"] * (1 - gt_df["total_movement"]) / 100)

clip_df = pd.read_csv(folder+"clip.csv")
clip_df = clip_df[["total_movement", "step"]]
#clip_df = smooth_and_downsample(clip_df)
clip_df["total_movement"] = clip_df["total_movement"].rolling(window=100).mean()
#clip_se = np.sqrt(clip_df["total_movement"] * (1 - clip_df["total_movement"]) / 100)

# Create Plotly figure
fig = go.Figure([])
# go.Scatter(
#         x=rlvlmf_df['step'],
#         y=rlvlmf_df['total_movement']- rlvlmf_se,
#         line=dict(color='red', width=0),
#         showlegend=False,
#         hoverinfo='skip'
#     ),
# go.Scatter(
#         x=rlvlmf_df['step'],
#         y=rlvlmf_df['total_movement'] + rlvlmf_se,
#         fill='tonexty',
#         fillcolor='rgba(255,0,0,0.2)',
#         line=dict(color='red', width=0),
#         name='Standard Error'
#     ),
# go.Scatter(
#         x=rlvlmf_wo_df['step'],
#         y=rlvlmf_wo_df['total_movement']- rlvlmf_wo_se,
#         line=dict(color='teal', width=0),
#         showlegend=False,
#         hoverinfo='skip'
#     ),
# go.Scatter(
#         x=rlvlmf_wo_df['step'],
#         y=rlvlmf_wo_df['total_movement'] + rlvlmf_wo_se,
#         fill='tonexty',
#         fillcolor='rgba(0,128,255,0.2)',
#         line=dict(color='teal', width=0),
#         name='Standard Error'
#     ),
# go.Scatter(
#         x=score_df['step'],
#         y=score_df['total_movement']- score_se,
#         line=dict(color='purple', width=0),
#         showlegend=False,
#         hoverinfo='skip'
#     ),
# go.Scatter(
#         x=score_df['step'],
#         y=score_df['total_movement'] + score_se,
#         fill='tonexty',
#         fillcolor='rgba(100,0,100,0.2)',
#         line=dict(color='purple', width=0),
#         name='Standard Error'
#     ),
# go.Scatter(
#         x=score_wo_df['step'],
#         y=score_wo_df['total_movement']- score_wo_se,
#         line=dict(color='orange', width=0),
#         showlegend=False,
#         hoverinfo='skip'
#     ),
# go.Scatter(
#         x=score_wo_df['step'],
#         y=score_wo_df['total_movement'] + score_wo_se,
#         fill='tonexty',
#         fillcolor='rgba(255,165,0,0.2)',
#         line=dict(color='orange', width=0),
#         name='Standard Error'
#     ),
# go.Scatter(
#         x=gt_df['step'],
#         y=gt_df['total_movement']- gt_se,
#         line=dict(color='green', width=0),
#         showlegend=False,
#         hoverinfo='skip'
#     ),
# go.Scatter(
#         x=gt_df['step'],
#         y=gt_df['total_movement'] + gt_se,
#         fill='tonexty',
#         fillcolor='rgba(0,255,0,0.2)',
#         line=dict(color='green', width=0),
#         name='Standard Error'
#     ),
# go.Scatter(
#         x=clip_df['step'],
#         y=clip_df['total_movement']- clip_se,
#         line=dict(color='blue', width=0),
#         showlegend=False,
#         hoverinfo='skip'
#     ),
# go.Scatter(
#         x=clip_df['step'],
#         y=clip_df['total_movement'] + clip_se,
#         fill='tonexty',
#         fillcolor='rgba(0,0,255,0.2)',
#         line=dict(color='blue', width=0),
#         name='Standard Error'
#     )

# RLVLMF line
fig.add_trace(go.Scattergl(
    x=rlvlmf_df['step'],
    y=rlvlmf_df['total_movement'],
    mode='lines',
    line=dict(color='orange', width=2),
    name='RLVLMF w/ metadata'
))

# VLM Score line
fig.add_trace(go.Scattergl(
    x=score_df['step'],
    y=score_df['total_movement'],
    mode='lines',
    line=dict(color='magenta', width=2),
    name='VLM Score w/ metadata'
))

# GT line
fig.add_trace(go.Scattergl(
    x=gt_df['step'],
    y=gt_df['total_movement'],
    mode='lines',
    line=dict(color='green', width=2),
    name='GT Preference'
))

# CLIP line
fig.add_trace(go.Scattergl(
    x=clip_df['step'],
    y=clip_df['total_movement'],
    mode='lines',
    line=dict(color='blue', width=2),
    name='CLIP Score'
))

fig.add_trace(go.Scattergl(
    x=rlvlmf_wo_df['step'],
    y=rlvlmf_wo_df['total_movement'],
    mode='lines',
    line=dict(color='red', width=2),
    name='RLVLMF w/o metadata'
))

fig.add_trace(go.Scattergl(
    x=score_wo_df['step'],
    y=score_wo_df['total_movement'],
    mode='lines',
    line=dict(color='purple', width=2),
    name='Score w/o metadata'
))

fig.update_layout(
    title='total_movement Over Time',
    xaxis=dict(
        title='Step',
        dtick=50000,
        range=[0,300000]
    ),
    # yaxis=dict(
    #     title='total_movement',
    #     dtick=.1,
    #     range=[0,1]
    # ),
    template='plotly_white'
)

# show and/or save figure
fig.show()
# fig.write_image("drawer_fig4.png")
#binary_data = fig.to_image('png')
#with open('image.png', 'wb') as f:
#    f.write(binary_data)
