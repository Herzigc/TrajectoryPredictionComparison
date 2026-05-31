import os
import json
import cv2
import numpy as np
import torch
import pandas as pd
from collections import deque
import time

# -------------------------
# SETTINGS
# -------------------------
past_len = 15
future_len = 90
num_keypoints = 17
width, height = 640, 480
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_folder = "../data/final/test_filled"
output_folder = f"../data/final/statistical_{past_len}_{future_len}"
os.makedirs(output_folder, exist_ok=True)

# Colors
model_colors = {
    "kalman": (0, 255, 0),
    "cv": (255, 255, 0)
}

# helper
def to_int_tuple(p):
    p = np.asarray(p).reshape(-1)

    # Handle NaNs / infs
    if not np.all(np.isfinite(p)):
        return (0, 0)

    return (int(p[0]), int(p[1]))

# -------------------------
# MODELS
# -------------------------
class KalmanCVModel:
    def __init__(self, future_len):
        self.future_len = future_len

    def eval(self):
        pass

    def __call__(self, x):
        centers = x[:, :, :2]
        B, T, _ = centers.shape
        preds = []
        dt = 1.0
        # State: [x, y, vx, vy]
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        Q = np.eye(4) * 1e-4  # process noise
        R = np.eye(2) * 1e-2  # measurement noise

        for b in range(B):
            traj = centers[b].cpu().numpy()

            # Initial state
            v0 = traj[-1] - traj[-2]
            x_state = np.array([traj[-1,0], traj[-1,1], v0[0], v0[1]])

            P = np.eye(4)

            # Run filter over past
            for t in range(T):
                z = traj[t]

                # Predict
                x_state = F @ x_state
                P = F @ P @ F.T + Q

                # Update
                y = z - H @ x_state
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.solve(S, np.eye(2))

                x_state = x_state + K @ y
                P = (np.eye(4) - K @ H) @ P

            # Predict future
            last_pos = x_state[:2].copy()
            future = []

            for _ in range(self.future_len):
                x_state = F @ x_state
                next_pos = x_state[:2]

                delta = next_pos - last_pos
                future.append(delta)
                last_pos = next_pos

            preds.append(future)

        return torch.from_numpy(np.array(preds, dtype=np.float32)).to(x.device)


class ConstantVelocityModel:
    def __init__(self, future_len):
        self.future_len = future_len

    def eval(self):
        pass

    def __call__(self, x):
        centers = x[:, :, :2]
        B, T, _ = centers.shape
        preds = []

        for b in range(B):
            traj = centers[b].cpu().numpy()

            velocities = traj[1:] - traj[:-1]
            v = velocities[-5:].mean(axis=0)

            last_pos = traj[-1].copy()
            future = []

            for _ in range(self.future_len):
                next_pos = last_pos + v
                delta = next_pos - last_pos
                future.append(delta)
                last_pos = next_pos

            preds.append(future)

        return torch.from_numpy(np.array(preds, dtype=np.float32)).to(x.device)


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
# INIT MODELS
# -------------------------
models = {
    "kalman": KalmanCVModel(future_len),
    "cv": ConstantVelocityModel(future_len)
}

# -------------------------
# PROCESS TEST DATA
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

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    writers = {}
    for name in models.keys():
        out_path = os.path.join(output_folder, f"{base}_{name}_pred_traj.mp4")
        writers[name] = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    buffer = deque(maxlen=past_len)
    full_gt_centers = []
    past_pred_centers = {name: deque(maxlen=1000) for name in models.keys()}

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
            continue

        kpts = np.array(ann["keypoints"][0])
        kpts[:, 0] /= width
        kpts[:, 1] /= height

        center_norm = center_from_kpts(kpts)
        center_px = center_norm * np.array([width, height])
        full_gt_centers.append(center_px)

        # Draw GT trajectory
        for i in range(1, len(full_gt_centers)):
            p1 = tuple(full_gt_centers[i - 1].astype(int))
            p2 = tuple(full_gt_centers[i].astype(int))
            cv2.line(frame, p1, p2, (204, 102, 255), 1)

        feat = np.concatenate([center_norm, kpts.flatten(), [ann["depth"]]])
        buffer.append(feat)

        if len(buffer) == past_len:
            x = torch.tensor(np.array(buffer), dtype=torch.float32, device=device)[None]
            last_center = center_norm

            for name, model in models.items():
                frame_model = frame.copy()

                # Timing the prediction
                t0 = time.perf_counter()
                deltas = model(x)[0].cpu().numpy()
                preds = np.cumsum(deltas, axis=0) + last_center
                inference_time = time.perf_counter() - t0
                model_times[name] += inference_time

                preds_px = preds * np.array([width, height])

                # Clamp predictions to prevent nans
                preds_px = np.nan_to_num(preds_px, nan=0.0, posinf=0.0, neginf=0.0)
                preds_px = np.clip(preds_px, [0, 0], [width, height])

                past_pred_centers[name].append({
                    "frame_idx": frame_idx,
                    "traj": preds_px
                })

                # Draw trajectory
                prev = np.array(buffer[-1][:2]) * np.array([width, height])
                for p in preds_px:
                    cv2.line(frame_model, to_int_tuple(prev), to_int_tuple(p), model_colors[name], 2)
                    prev = p

                writers[name].write(frame_model)

        frame_idx += 1

    cap.release()
    for w in writers.values():
        w.release()

    # -------------------------
    # METRICS
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
# EXPORT
# -------------------------
df = pd.DataFrame(results)
csv_path = os.path.join(output_folder, "statistical_results.csv")
df.to_csv(csv_path, index=False)

print("Saved:", csv_path)
print("\nMean metrics:")
print(df.groupby("model")[["ADE_px", "FDE_px"]].mean())