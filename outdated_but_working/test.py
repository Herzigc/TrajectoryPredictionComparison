import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
from collections import deque

# -------------------------
# SETTINGS
# -------------------------
past_len = 15
future_len = 60
num_keypoints = 17
width, height = 640, 480
input_dim = 2 + num_keypoints * 2 + 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_folder = "../data/final/test_filled"
output_folder = f"../data/final/{past_len}_{future_len}"
os.makedirs(output_folder, exist_ok=True)

# Colors
model_colors = {
    "lstm": (0, 0, 255),
    "gru": (0, 255, 255),
    "transformer": (255, 0, 0)
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
        out, _ = self.lstm(x)
        out = out[:, -1]
        out = self.fc(out)
        return out.view(-1, future_len, 2)


class GRUModel(nn.Module):
    def __init__(self, input_dim, future_len):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, future_len * 2)

    def forward(self, x):
        out, _ = self.gru(x)
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
        attn, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.fc_out(x)
        return x.view(-1, future_len, 2)


# -------------------------
# LOAD MODELS
# -------------------------
model_paths = {
    "lstm": f"lstm_{past_len}_{future_len}.pt",
    "gru": f"gru_{past_len}_{future_len}.pt",
    "transformer": f"transformer_{past_len}_{future_len}.pt"
}

models = {}
for name, path in model_paths.items():
    model = torch.load(path, map_location=device, weights_only=False)
    model.recursive = True
    model.eval()
    models[name] = model


# -------------------------
# UTILS
# -------------------------
def center_from_kpts(kpts):
    return (kpts[15] + kpts[16]) / 2.0


def draw_dashed_line(img, p1, p2, color, thickness=2, dash_len=30):
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-2:
        return
    n = max(int(dist // dash_len), 1)
    for i in range(0, n, 2):
        t0 = i / n
        t1 = min((i + 1) / n, 1.0)
        s = tuple(np.round(p1 + t0 * (p2 - p1)).astype(int))
        e = tuple(np.round(p1 + t1 * (p2 - p1)).astype(int))
        cv2.line(img, s, e, color, thickness)



# -------------------------
# PROCESS EACH TEST SEQUENCE
# -------------------------
results = []

for f in os.listdir(test_folder):
    if not f.endswith("_rgxb.json"):
        continue
    base = f.replace("_rgxb.json", "")
    video_path = os.path.join(test_folder, base + "_rgxb.mp4")
    json_path = os.path.join(test_folder, f)

    with open(json_path) as jf:
        data = json.load(jf)

    # Prepare video writers for each model
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    writers = {}
    for name in models.keys():
        out_path = os.path.join(output_folder, f"{base}_{name}_pred_traj.mp4")
        writers[name] = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    buffer = deque(maxlen=past_len)
    full_gt_centers = []
    past_pred_centers = {name: deque(maxlen=1000) for name in models.keys()}
    max_frame = max(a["frame_number"] for a in data["annotations"])
    frame_idx = 1

    # Timer dictionary
    model_times = {name: 0.0 for name in models.keys()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ann = next((a for a in data["annotations"] if a["frame_number"] == frame_idx), None)
        if ann is None:
            frame_idx += 1
            print("krise")
            continue

        kpts = np.array(ann["keypoints"][0])
        kpts[:, 0] /= width
        kpts[:, 1] /= height
        center_norm = center_from_kpts(kpts)
        depth = ann["depth"]
        center_px = center_norm * np.array([width, height])
        full_gt_centers.append(center_px)

        n = len(full_gt_centers)
        for i in range(1, n):
            p1 = tuple(full_gt_centers[i - 1].astype(int))
            p2 = tuple(full_gt_centers[i].astype(int))

            if i <= n - past_len:  # forgotten GT
                draw_dashed_line(frame, p1, p2, (0, 255, 0), 1)
            else:  # known GT
                cv2.line(frame, p1, p2, (204, 102, 255), 1)

        feat = np.concatenate([center_norm, kpts.flatten(), [depth]])
        buffer.append(feat)

        # Predict only if we have enough past frames and enough GT to compare
        if len(buffer) == past_len:
            x = torch.tensor(np.array(buffer), dtype=torch.float32, device=device)[None]
            last_center = center_norm
            for name, model in models.items():
                frame_model = frame.copy()

                # Timing the prediction
                t0 = time.time()
                deltas = model(x)[0].detach().cpu().numpy()
                preds = np.cumsum(deltas, axis=0) + last_center
                inference_time = time.time() - t0
                model_times[name] += inference_time

                preds_px = preds * np.array([width, height])
                past_pred_centers[name].append({
                    "frame_idx": frame_idx,
                    "traj": preds_px
                })

                pp = past_pred_centers[name]
                for i in range(1, len(pp)):
                    # Only draw predictions older than past_len
                    if pp[i - 1]["frame_idx"] <= frame_idx - past_len:
                        p_prev = tuple(pp[i - 1]["traj"][0].astype(int))
                        p_curr = tuple(pp[i]["traj"][0].astype(int))
                        draw_dashed_line(frame_model, p_prev, p_curr, model_colors[name], 1)

                # Draw predicted future trajectory as solid line
                prev = (buffer[-1][:2] * np.array([width, height])).astype(int)
                for p in preds_px:
                    cv2.line(frame_model, tuple(prev.astype(int)), tuple(p.astype(int)), model_colors[name], 1)
                    prev = p

                writers[name].write(frame_model)

        frame_idx += 1

    cap.release()
    for w in writers.values():
        w.release()

    # -------------------------
    # ADE / FDE METRICS
    # -------------------------
    for name in models.keys():
        all_ade, all_fde = [], []

        for item in past_pred_centers[name]:
            t0 = item["frame_idx"] - 1
            traj = item["traj"]

            if t0 + future_len > len(full_gt_centers):
                continue

            gt = np.array(full_gt_centers[t0: t0 + future_len])
            pr = np.array(traj)

            if gt.shape != pr.shape:
                continue

            dists = np.linalg.norm(gt - pr, axis=1)
            all_ade.append(dists.mean())
            all_fde.append(dists[-1])

        results.append({
            "sequence": base,
            "model": name,
            "ADE_px": np.mean(all_ade) if all_ade else np.nan,
            "FDE_px": np.mean(all_fde) if all_fde else np.nan,
            "runtime_sec": model_times[name]
        })

# -------------------------
# EXPORT RESULTS
# -------------------------
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_folder, "nar_displacement_results.csv"), index=False)
print("Mean metrics:")
print(df.groupby("model")[["ADE_px", "FDE_px"]].mean())
