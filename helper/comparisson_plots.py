import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, f_oneway
import matplotlib.colors as mcolors


brightness_levels = [0.8, 1, 1.2, 1.4, 1.6, 1.8]
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


base_colors = {
    "lstm": "#0072B2",        # blue
    "gru": "#E69F00",         # orange
    "transformer": "#009E73", # green
    "kalman": "#D55E00",      # red/orange
    "cv": "#CC79A7"           # purple
}


def adjust_color(color, factor):
    c = np.array(mcolors.to_rgb(color))

    if factor < 1:
        # darker
        return tuple(np.clip(c * factor, 0, 1))
    else:
        # lighter (blend toward white)
        return tuple(np.clip(c + (1 - c) * (factor - 1), 0, 1))

def add_jitter(ax, data, positions, size, alpha):
    for i, vals in enumerate(data):
        x = np.random.normal(positions[i], 0.04, size=len(vals))  # small horizontal jitter
        ax.scatter(
            x,
            vals,
            color="black",
            alpha=alpha,
            s=size,
            zorder=3
        )

def add_percent_axis(ax):
    diag = np.sqrt(640 ** 2 + 480 ** 2)

    secax = ax.secondary_yaxis(
        'right',
        functions=(
            lambda x: (x / diag) * 100,  # px -> %
            lambda x: (x / 100) * diag  # % -> px
        )
    )

    secax.set_ylabel("Error %", fontsize=10)

    return secax

# =========================
# STATISTICS FUNCTION
# =========================

def compute_stats(subset, metric):
    subset_agg = (
        subset
        .groupby(["sequence", "model"], as_index=False)[metric]
        .mean()
    )

    pivot = subset_agg.pivot(index="sequence", columns="model", values=metric)

    pivot = pivot.dropna()

    models = pivot.columns
    values = [pivot[m] for m in models]

    # Friedman
    try:
        _, p_friedman = friedmanchisquare(*values)
    except:
        p_friedman = np.nan

    # ANOVA
    try:
        _, p_anova = f_oneway(*values)
    except:
        p_anova = np.nan

    return p_friedman, p_anova


# =========================
# PLOTTING FUNCTION
# =========================

def plot_config(config=None, past=None, future=None):
    if config is not None:
        subset = df_all[df_all["config"] == config]
        title_cfg = config
        filename = f"plot_{config}.png"

    else:
        subset = df_all.copy()

        if past is not None:
            subset = subset[subset["past_len"] == past]

        if future is not None:
            subset = subset[subset["future_len"] == future]

        if past is not None and future is not None:
            title_cfg = f"P{past}_F{future}"
            filename = f"plot_P{past}_F{future}.png"

        elif past is not None:
            title_cfg = f"P{past}_ALL_FUTURES"
            filename = f"plot_P{past}_all_futures.png"

        elif future is not None:
            title_cfg = f"ALL_PAST_F{future}"
            filename = f"plot_all_past_F{future}.png"

        else:
            title_cfg = "ALL CONFIGS"
            filename = "plot_all.png"
    fig, axes = plt.subplots(1, 3, figsize=(18,13))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        data = []
        labels = []

        for model in model_order:
            vals = subset[subset["model"] == model][metric]
            data.append(vals)
            labels.append(model)

        p_friedman, p_anova = compute_stats(subset, metric)
        #print(p_friedman, p_anova) -> this is way below 0.001 hence why its not in the label directly

        positions = list(range(1, len(data) + 1))

        bp = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1)
        )

        # colors
        for patch, model in zip(bp["boxes"], model_order):
            patch.set_facecolor(base_colors.get(model, "#cccccc"))

        # jitter
        if config is None and past is None and future is None:
            add_jitter(ax, data, positions, 5, 0.125)
        else:
            add_jitter(ax, data, positions, 7, 0.25)

        # labels
        label_map = {
            "ADE_px": "ADE (px)",
            "FDE_px": "FDE (px)",
            "runtime_sec": "Runtime (s)"
        }

        ax.set_ylabel(
            f"{label_map[metric]}\n(ANOVA & Friedman p < 0.001)",
            fontsize=11
        )

        if metric == "ADE_px":
            ax.set_ylim(0, 100)
            add_percent_axis(ax)

        elif metric == "FDE_px":
            ax.set_ylim(0, 140)
            add_percent_axis(ax)

        else:
            ax.set_ylim(0, 2)

        ax.set_xticklabels(labels, rotation=45)

        ax.grid(True)

    fig.suptitle(f"{title_cfg} — Model Comparison", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"./images/{filename}", dpi=300)
    #plt.show()

# -------------------------
# GLOBAL PLOT (ALL CONFIGS)
# -------------------------
def plot_all_configs():
    fig, axes = plt.subplots(3, 1, figsize=(18, 13))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        data = []
        labels = []
        colors = []

        for ci, config in enumerate(config_order):
            subset_cfg = df_all[df_all["config"] == config]

            factor = brightness_levels[ci % len(brightness_levels)]

            for model in model_order:
                vals = subset_cfg[subset_cfg["model"] == model][metric]

                data.append(vals)
                labels.append(f"{model}\n{config}")

                base = base_colors.get(model, "#999999")
                colors.append(adjust_color(base, factor))

        p_friedman, p_anova = compute_stats(df_all, metric)
        # print(p_friedman, p_anova)

        positions = list(range(1, len(data) + 1))

        bp = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1)
        )

        # colors
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)

        add_jitter(ax, data, positions, 5, 0.125)

        label_map = {
            "ADE_px": "ADE (px)",
            "FDE_px": "FDE (px)",
            "runtime_sec": "Runtime (s)"
        }

        ax.set_ylabel(
            f"{label_map[metric]}\n(p < 0.001)",
            fontsize=11
        )

        if metric in ["ADE_px"]:
            ax.set_ylim(0, 100)
            add_percent_axis(ax)
        elif metric in ["FDE_px"]:
            ax.set_ylim(0, 140)
            add_percent_axis(ax)
        else:
            ax.set_ylim(0, 2)

        if i == len(metrics) - 1:
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticklabels([])

        ax.grid(True, alpha=0.3)

    fig.suptitle("All Configurations — Model Comparison", fontsize=16)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    plt.savefig("./images/plot_all_configs.png", dpi=300)
    #plt.show()

# =========================
# GENERATE ALL PLOTS
# =========================

# per config
for cfg in config_order:
    plot_config(cfg)

# 5 plots for combined stuff
plot_config(past=15)
plot_config(past=30)
plot_config(future=30)
plot_config(future=60)
plot_config(future=90)

# global all configs med
plot_config(None)

# global all configs separate
plot_all_configs()

def plot_all_past():
    past_values = sorted(df_all["past_len"].unique())

    # ======================
    # 2x2 (ADE + FDE)
    # ======================
    fig, axes = plt.subplots(2, len(past_values), figsize=(14, 14))

    metric_map = ["ADE_px", "FDE_px"]

    for col, past in enumerate(past_values):
        subset = df_all[df_all["past_len"] == past]

        for row, metric in enumerate(metric_map):
            ax = axes[row, col]

            data = []
            for model in model_order:
                vals = subset[subset["model"] == model][metric]
                data.append(vals)

            positions = range(1, len(data) + 1)

            bp = ax.boxplot(
                data,
                positions=positions,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1)
            )

            for patch, model in zip(bp["boxes"], model_order):
                patch.set_facecolor(base_colors[model])

            add_jitter(ax, data, positions, 4, 0.15)

            ax.set_title(f"{metric.replace('_px','')} - P{past}")
            ax.set_xticklabels(model_order, rotation=45)

            if metric == "ADE_px":
                ax.set_ylim(0, 100)
                add_percent_axis(ax)
            else:
                ax.set_ylim(0, 140)
                add_percent_axis(ax)

            ax.grid(True, alpha=0.3)

    fig.suptitle("ADE & FDE grouped by PAST", fontsize=14)
    plt.tight_layout()
    plt.savefig("./images/grouped_past_ADE_FDE.png", dpi=300)
    plt.show()


    # ======================
    # 1x2 (Runtime)
    # ======================
    fig, axes = plt.subplots(1, len(past_values), figsize=(14, 12))

    for col, past in enumerate(past_values):
        ax = axes[col]
        subset = df_all[df_all["past_len"] == past]

        data = []
        for model in model_order:
            vals = subset[subset["model"] == model]["runtime_sec"]
            data.append(vals)

        positions = range(1, len(data) + 1)

        bp = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1)
        )

        for patch, model in zip(bp["boxes"], model_order):
            patch.set_facecolor(base_colors[model])

        add_jitter(ax, data, positions, 4, 0.15)

        ax.set_title(f"Runtime - P{past}")
        ax.set_xticklabels(model_order, rotation=45)
        ax.set_ylim(0, 2)

        ax.grid(True, alpha=0.3)

    fig.suptitle("Runtime grouped by PAST", fontsize=14)
    plt.tight_layout()
    plt.savefig("./images/grouped_past_runtime.png", dpi=300)
    plt.show()

def plot_all_future():
    future_values = sorted(df_all["future_len"].unique())

    # ======================
    # 2x3 (ADE + FDE)
    # ======================
    fig, axes = plt.subplots(2, len(future_values), figsize=(14, 12))

    metric_map = ["ADE_px", "FDE_px"]

    for col, future in enumerate(future_values):
        subset = df_all[df_all["future_len"] == future]

        for row, metric in enumerate(metric_map):
            ax = axes[row, col]

            data = []
            for model in model_order:
                vals = subset[subset["model"] == model][metric]
                data.append(vals)

            positions = range(1, len(data) + 1)

            bp = ax.boxplot(
                data,
                positions=positions,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1)
            )

            for patch, model in zip(bp["boxes"], model_order):
                patch.set_facecolor(base_colors[model])

            add_jitter(ax, data, positions, 4, 0.15)

            ax.set_title(f"{metric.replace('_px','')} - F{future}")
            ax.set_xticklabels(model_order, rotation=45)

            if metric == "ADE_px":
                ax.set_ylim(0, 100)
                add_percent_axis(ax)
            else:
                ax.set_ylim(0, 140)
                add_percent_axis(ax)

            ax.grid(True, alpha=0.3)

    fig.suptitle("ADE & FDE grouped by FUTURE", fontsize=14)
    plt.tight_layout()
    plt.savefig("./images/grouped_future_ADE_FDE.png", dpi=300)
    plt.show()


    # ======================
    # 1x3 (Runtime)
    # ======================
    fig, axes = plt.subplots(1, len(future_values), figsize=(14, 12))

    for col, future in enumerate(future_values):
        ax = axes[col]
        subset = df_all[df_all["future_len"] == future]

        data = []
        for model in model_order:
            vals = subset[subset["model"] == model]["runtime_sec"]
            data.append(vals)

        positions = range(1, len(data) + 1)

        bp = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1)
        )

        for patch, model in zip(bp["boxes"], model_order):
            patch.set_facecolor(base_colors[model])

        add_jitter(ax, data, positions, 4, 0.15)

        ax.set_title(f"Runtime - F{future}")
        ax.set_xticklabels(model_order, rotation=45)
        ax.set_ylim(0, 2)

        ax.grid(True, alpha=0.3)

    fig.suptitle("Runtime grouped by FUTURE", fontsize=14)
    plt.tight_layout()
    plt.savefig("./images/grouped_future_runtime.png", dpi=300)
    plt.show()


plot_all_past()
plot_all_future()