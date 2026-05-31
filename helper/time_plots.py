import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# -------------------------
# SETTINGS
# -------------------------
sequence_folder = "../data/final/test_filled"
plot_folder = "../data/final/plots"
os.makedirs(plot_folder, exist_ok=True)

past_len = 15
future_len = 60
num_keypoints = 17
width, height = 640, 480

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# MODELS
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, future_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)

        self.attn_pool = nn.Linear(128, 1)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, future_len * 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, 128)

        weights = torch.softmax(self.attn_pool(out), dim=1)  # (B, T, 1)
        context = (out * weights).sum(dim=1)  # (B, 128)

        out = self.fc(context)
        return out.view(-1, future_len, 2)

class GRUModel(nn.Module):
    def __init__(self, input_dim, future_len):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, batch_first=True)

        self.attn_pool = nn.Linear(128, 1)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, future_len * 2)
        )

    def forward(self, x):
        out, _ = self.gru(x)

        weights = torch.softmax(self.attn_pool(out), dim=1)
        context = (out * weights).sum(dim=1)

        out = self.fc(context)
        return out.view(-1, future_len, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, future_len):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, 128)
        self.pos_enc = PositionalEncoding(128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.attn_pool = nn.Linear(128, 1)

        self.fc_out = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, future_len * 2)
        )

    def forward(self, x):
        x = self.fc_in(x)
        x = self.pos_enc(x)

        x = self.encoder(x)  # (B, T, 128)

        weights = torch.softmax(self.attn_pool(x), dim=1)
        context = (x * weights).sum(dim=1)

        out = self.fc_out(context)
        return out.view(-1, future_len, 2)



# -------------------------
# LOAD MODELS
# -------------------------
model_paths = {
    "gru": f"../gru_{past_len}_{future_len}_new.pt",
    "lstm": f"../lstm_{past_len}_{future_len}_new.pt",
    "transformer": f"../transformer_{past_len}_{future_len}_new.pt"
}

models = {}
for name, path in model_paths.items():
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    models[name] = model


# -------------------------
# UTILS
# -------------------------
def center_from_kpts(kpts):
    return (kpts[15] + kpts[16]) / 2.0

def load_sequence(json_path):
    with open(json_path) as f:
        data = json.load(f)

    centers = []
    feats = []

    prev_center = None
    prev_velocity = np.zeros(2)

    for ann in data["annotations"]:
        kpts = np.array(ann["keypoints"][0])
        kpts[:, 0] /= width
        kpts[:, 1] /= height

        center = center_from_kpts(kpts)
        depth = ann["depth"]

        if prev_center is None:
            vel = np.zeros(2)
            acc = np.zeros(2)
        else:
            vel = center - prev_center
            acc = vel - prev_velocity

        feat = np.concatenate([
            center,
            vel,
            acc,
            kpts.flatten(),
            [depth]
        ])

        centers.append(center)
        feats.append(feat)

        prev_center = center
        prev_velocity = vel

    return np.array(centers), np.array(feats)


# -------------------------
# PLOT
# -------------------------
def plot_random(sequence_name):

    json_path = os.path.join(sequence_folder, f"{sequence_name}_rgxb.json")
    centers, feats = load_sequence(json_path)

    if len(centers) <= past_len + future_len:
        raise ValueError("Sequence too short")

    start = random.randint(past_len, len(centers) - future_len - 1)

    past = centers[start - past_len:start]
    gt = centers[start:start + future_len]

    past_px = past * np.array([width, height])
    gt_px = gt * np.array([width, height])

    last = centers[start - 1]

    # -------------------------
    # FIGURE SETUP
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    model_order = ["gru", "lstm", "transformer"]

    model_colors = {
        "gru": "gold",
        "lstm": "red",
        "transformer": "blue"
    }

    memory_color = "hotpink"

    # -------------------------
    # LOOP MODELS
    # -------------------------
    for ax, name in zip(axes, model_order):

        model = models[name]

        x = torch.tensor(
            feats[start - past_len:start],
            dtype=torch.float32,
            device=device
        )[None]

        raw = model(x)[0].detach().cpu().numpy()
        pred = np.cumsum(raw, axis=0) + last
        pred_px = pred * np.array([width, height])

        # -------------------------
        # ADE / FDE
        # -------------------------
        dists = np.linalg.norm(gt_px - pred_px, axis=1)
        ade_px = dists.mean()
        fde_px = dists[-1]

        # -------------------------
        # MEMORY
        # -------------------------
        ax.plot(
            past_px[:, 0],
            past_px[:, 1],
            color=memory_color,
            linewidth=2,
            label=f"Memory ({past_len})"
        )

        # -------------------------
        # GT
        # -------------------------
        ax.plot(
            gt_px[:, 0],
            gt_px[:, 1],
            color="limegreen",
            linewidth=2,
            label=f"Ground Truth ({future_len})"
        )

        ax.plot(
            [past_px[-1, 0], gt_px[0, 0]],
            [past_px[-1, 1], gt_px[0, 1]],
            color="limegreen",
            linewidth=2,
            alpha=0.6
        )

        # -------------------------
        # Predictions
        # -------------------------
        ax.plot(
            pred_px[:, 0],
            pred_px[:, 1],
            color=model_colors[name],
            linewidth=2,
            label="Prediction over time"
        )

        # -------------------------
        # AXIS FORMATTING
        # -------------------------
        ax.set_title(
            f"{name.upper()} | "
            f"ADE: {ade_px:.1f}px ({(ade_px / np.sqrt(width ** 2 + height ** 2)) * 100:.2f}%) | "
            f"FDE: {fde_px:.1f}px ({(fde_px / np.sqrt(width ** 2 + height ** 2)) * 100:.2f}%)"
        )

        ax.set_xlabel("X position (px)")
        ax.set_ylabel("Y position (px)")

    # -------------------------
    # GLOBAL LEGEND
    # -------------------------
    handles = [
        plt.Line2D([], [], color="hotpink", label=f"Memory ({past_len} frames)"),
        plt.Line2D([], [], color="limegreen", label=f"Ground Truth ({future_len} frames)"),
        plt.Line2D([], [], color="red", label=f"LSTM prediction over time ({future_len} frames)"),
        plt.Line2D([], [], color="yellow", label=f"GRU prediction over time ({future_len} frames)"),
        plt.Line2D([], [], color="blue", label=f"Transformer prediction over time ({future_len} frames)"),
    ]

    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    out = os.path.join(plot_folder, f"{sequence_name}_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    plot_random("clemens_seq09")