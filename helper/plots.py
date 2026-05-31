import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import random

# -------------------------
# SETTINGS
# -------------------------
plot_folder = "../data/final/plots"
os.makedirs(plot_folder, exist_ok=True)
sequence_folder = "../data/final/test_filled"  # folder with JSONs
past_len = 15
future_len = 30
num_keypoints = 17
width, height = 640, 480
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models paths
model_paths = {
    "lstm": f"lstm_{past_len}_{future_len}.pt",
    "gru": f"gru_{past_len}_{future_len}.pt",
    "transformer": f"transformer_{past_len}_{future_len}.pt"
}

# -------------------------
# MODELS
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, future_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, future_len * 2)
    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:, -1]
        out = self.fc(out)
        return out.view(-1, future_len, 2)

class GRUModel(nn.Module):
    def __init__(self, input_dim, future_len):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, future_len * 2)
    def forward(self, x):
        out,_ = self.gru(x)
        out = out[:, -1]
        out = self.fc(out)
        return out.view(-1, future_len, 2)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, future_len):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 64)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(64, future_len * 2)
    def forward(self, x):
        x = self.fc_in(x)
        attn,_ = self.attn(x,x,x,need_weights=False)
        x = self.norm(x + attn)
        x = x.transpose(1,2)
        x = self.pool(x).squeeze(-1)
        x = self.fc_out(x)
        return x.view(-1, future_len, 2)

# -------------------------
# UTILS
# -------------------------
def center_from_kpts(kpts):
    return (kpts[15] + kpts[16]) / 2.0

# Load models
input_dim = 2 + num_keypoints*2 + 1
models = {}
for name, path in model_paths.items():
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    models[name] = model

# -------------------------
# HELPER TO LOAD SEQUENCE
# -------------------------
def load_sequence(json_path):
    with open(json_path) as jf:
        data = json.load(jf)
    centers, feats = [], []
    for ann in data["annotations"]:
        kpts = np.array(ann["keypoints"][0])
        kpts[:,0] /= width
        kpts[:,1] /= height
        center = center_from_kpts(kpts)
        depth = ann["depth"]
        feat = np.concatenate([center, kpts.flatten(), [depth]])
        centers.append(center)
        feats.append(feat)
    return np.array(centers), np.array(feats)

# -------------------------
# PLOTTING FUNCTION
# -------------------------
def plot_random_frame(sequence_name):
    json_path = os.path.join(sequence_folder, f"{sequence_name}_rgxb.json")
    centers, feats = load_sequence(json_path)
    num_frames = len(centers)
    if num_frames <= past_len + future_len:
        raise ValueError("Sequence too short for past+future lengths")

    # Pick a random frame where we can predict
    start_idx = random.randint(past_len, num_frames - future_len - 1)
    past_coords = centers[start_idx - past_len: start_idx]
    gt_future = centers[start_idx: start_idx + future_len]

    # Convert past/GT to pixels
    past_px = past_coords * np.array([width, height])
    gt_px   = gt_future * np.array([width, height])

    # Plot each model
    for name, model in models.items():
        x = torch.tensor(feats[start_idx - past_len: start_idx], dtype=torch.float32, device=device)[None]
        pred_deltas = model(x)[0].detach().cpu().numpy()
        pred_future = np.cumsum(pred_deltas, axis=0) + centers[start_idx - 1]

        # Convert predicted future to pixels
        pred_px = pred_future * np.array([width, height])

        # Compute ADE/FDE in pixels
        dists = np.linalg.norm(gt_px - pred_px, axis=1)
        ade = dists.mean()
        fde = dists[-1]

        # Compute ADE/FDE in pixels (split X/Y)
        ade_x_px = np.mean(np.abs(gt_px[:, 0] - pred_px[:, 0]))
        ade_y_px = np.mean(np.abs(gt_px[:, 1] - pred_px[:, 1]))
        fde_x_px = np.abs(gt_px[-1, 0] - pred_px[-1, 0])
        fde_y_px = np.abs(gt_px[-1, 1] - pred_px[-1, 1])

        # Compute ADE/FDE in %
        ade_x_pct = ade_x_px / width * 100
        ade_y_pct = ade_y_px / height * 100
        fde_x_pct = fde_x_px / width * 100
        fde_y_pct = fde_y_px / height * 100

        # Plot X coordinates in pixels
        plt.figure(figsize=(12,4))
        plt.plot(range(-past_len,0), past_px[:,0],'b-o', label="Past knowledge (X)")
        plt.plot(range(0,future_len), gt_px[:,0],'g-o', label="GT future (X)")
        plt.plot(range(0,future_len), pred_px[:,0],'r-o', label="Predicted future (X)")
        plt.xlabel("Frames relative to prediction start")
        plt.ylabel("X coordinate (px)")
        plt.title(f"{name.upper()} | "f"ADE: {ade_x_px:.2f}px {ade_x_pct:.2f}% | "f"FDE: {fde_x_px:.2f}px {fde_x_pct:.2f}%")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(plot_folder, f"{name}_frame{start_idx}_X.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

        # Plot Y coordinates in pixels
        plt.figure(figsize=(12,4))
        plt.plot(range(-past_len,0), past_px[:,1],'b-o', label="Past knowledge (Y)")
        plt.plot(range(0,future_len), gt_px[:,1],'g-o', label="GT future (Y)")
        plt.plot(range(0,future_len), pred_px[:,1],'r-o', label="Predicted future (Y)")
        plt.xlabel("Frames relative to prediction start")
        plt.ylabel("Y coordinate (px)")
        plt.title(f"{name.upper()} | "f"ADE: {ade_y_px:.2f}px {ade_y_pct:.2f}% | "f"FDE: {fde_y_px:.2f}px {fde_y_pct:.2f}%")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(plot_folder, f"{name}_frame{start_idx}_Y.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    # Replace this with your sequence
    seq_name = "clemens_seq09"
    plot_random_frame(seq_name)
