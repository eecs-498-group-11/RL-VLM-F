import pandas as pd

# === CONFIGURATION ===
input_file = "exp/soccer_3_images/metaworld_soccer-v2/2025-11-11-12-01-45/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch40_Rupdate5_en3_sample0_large_batch10_seed0/train.csv"       # Path to your CSV file
output_file = "exp/soccer_3_images/metaworld_soccer-v2/2025-11-11-12-01-45/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch40_Rupdate5_en3_sample0_large_batch10_seed0/train_mod.csv"     # Path to save the modified file
column_to_modify = "step"    # Name of the column you want to add 50k to
amount_to_add = 200000          # Amount to add

# === LOAD CSV ===
df = pd.read_csv(input_file)

# === MODIFY COLUMN ===
if column_to_modify in df.columns:
    df[column_to_modify] = df[column_to_modify] + amount_to_add
else:
    raise ValueError(f"Column '{column_to_modify}' not found in CSV.")

# === SAVE RESULT ===
df.to_csv(output_file, index=False)

print(f"Updated CSV saved to {output_file}")
