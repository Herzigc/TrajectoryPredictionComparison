import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np

# =========================
# LOAD DATA
# =========================

def load_pair(stats_path, ml_path, past, future):
    df_stats = pd.read_csv(stats_path)
    df_ml = pd.read_csv(ml_path)

    df = pd.concat([df_stats, df_ml], ignore_index=True)
    df["past_len"] = past
    df["future_len"] = future
    return df


df_all = pd.concat([
    load_pair("../data/csvs/15_30_stats.csv", "../data/csvs/15_30_ml_new.csv", 15, 30),
    load_pair("../data/csvs/15_60_stats.csv", "../data/csvs/15_60_ml_new.csv", 15, 60),
    load_pair("../data/csvs/15_90_stats.csv", "../data/csvs/15_90_ml_new.csv", 15, 90),
    load_pair("../data/csvs/30_30_stats.csv", "../data/csvs/30_30_ml_new.csv", 30, 30),
    load_pair("../data/csvs/30_60_stats.csv", "../data/csvs/30_60_ml_new.csv", 30, 60),
    load_pair("../data/csvs/30_90_stats.csv", "../data/csvs/30_90_ml_new.csv", 30, 90)
], ignore_index=True)

# config label
df_all["config"] = "P" + df_all["past_len"].astype(str) + "_F" + df_all["future_len"].astype(str)

metrics = ["ADE_px", "FDE_px", "runtime_sec"]
model_order = ["cv", "kalman", "lstm", "gru", "transformer"]
config_order = sorted(df_all["config"].unique())

# === GLOBAL SUMMARY TABLE ===

summary_all = df_all.groupby("model")[metrics].mean().reset_index()

# rename
summary_all = summary_all.rename(columns={
    "ADE_px": "mean_ADE",
    "FDE_px": "mean_FDE",
    "runtime_sec": "mean_speed"
})

# round
summary_all["mean_ADE"] = summary_all["mean_ADE"].round(3)
summary_all["mean_FDE"] = summary_all["mean_FDE"].round(3)
summary_all["mean_speed"] = summary_all["mean_speed"].round(5)

# enforce model order
summary_all["model"] = pd.Categorical(summary_all["model"], categories=model_order, ordered=True)
summary_all = summary_all.sort_values("model")

print("\n=== GLOBAL MODEL PERFORMANCE SUMMARY ===\n")
print(summary_all.to_string(index=False))


summary_std = df_all.groupby("model")[metrics].std().reset_index()

summary_std = summary_std.rename(columns={
    "ADE_px": "std_ADE",
    "FDE_px": "std_FDE",
    "runtime_sec": "std_speed"
})

summary_std = summary_std.round(3)

# enforce model order
summary_std["model"] = pd.Categorical(summary_std["model"], categories=model_order, ordered=True)
summary_std = summary_std.sort_values("model")


print("\n=== MODEL VARIABILITY (STD ACROSS CONFIGS) ===\n")
print(summary_std.to_string(index=False))

final_table = summary_all.merge(summary_std, on="model")

# =========================
# SUMMARY BY PAST LENGTH (aggregated over ALL futures)
# =========================

summary_past_mean = df_all.groupby(["past_len", "model"])[metrics].mean().reset_index()
summary_past_std  = df_all.groupby(["past_len", "model"])[metrics].std().reset_index()

summary_past_mean = summary_past_mean.rename(columns={
    "ADE_px": "mean_ADE",
    "FDE_px": "mean_FDE",
    "runtime_sec": "mean_speed"
})

summary_past_std = summary_past_std.rename(columns={
    "ADE_px": "std_ADE",
    "FDE_px": "std_FDE",
    "runtime_sec": "std_speed"
})

summary_past = summary_past_mean.merge(summary_past_std, on=["past_len", "model"])

# formatting
summary_past = summary_past.round(3)
summary_past["model"] = pd.Categorical(summary_past["model"], categories=model_order, ordered=True)
summary_past = summary_past.sort_values(["past_len", "model"])

print("\n=== PERFORMANCE BY PAST LENGTH (averaged over all futures) ===\n")
print(summary_past.to_string(index=False))


# =========================
# SUMMARY BY FUTURE LENGTH (aggregated over ALL pasts)
# =========================

summary_future_mean = df_all.groupby(["future_len", "model"])[metrics].mean().reset_index()
summary_future_std  = df_all.groupby(["future_len", "model"])[metrics].std().reset_index()

summary_future_mean = summary_future_mean.rename(columns={
    "ADE_px": "mean_ADE",
    "FDE_px": "mean_FDE",
    "runtime_sec": "mean_speed"
})

summary_future_std = summary_future_std.rename(columns={
    "ADE_px": "std_ADE",
    "FDE_px": "std_FDE",
    "runtime_sec": "std_speed"
})

summary_future = summary_future_mean.merge(summary_future_std, on=["future_len", "model"])

# formatting
summary_future = summary_future.round(3)
summary_future["model"] = pd.Categorical(summary_future["model"], categories=model_order, ordered=True)
summary_future = summary_future.sort_values(["future_len", "model"])

print("\n=== PERFORMANCE BY FUTURE LENGTH (averaged over all pasts) ===\n")
print(summary_future.to_string(index=False))