import pandas as pd

df1 = pd.read_csv("eval.csv")
df2 = pd.read_csv("eval_1.csv")

# Last non-zero episode and step from df1
last_ep = df1.loc[df1["episode"] != 0, "episode"].max()
last_step = df1.loc[df1["step"] != 0, "step"].max()

# --- EPISODES (episode 0 stays 0) ---
df2.loc[df2["episode"] != 0, "episode"] += last_ep

# --- STEPS (continuous, exactly +500 gap) ---
df2_nonzero_min = df2.loc[df2["step"] != 0, "step"].min()

# Offset so that the FIRST non-zero step in df2 becomes last_step + 500
step_offset = (last_step + 500) - df2_nonzero_min

df2.loc[df2["step"] != 0, "step"] += step_offset

# Combine
combined = pd.concat([df1, df2], ignore_index=True)
combined.to_csv("eval_combined.csv", index=False)
