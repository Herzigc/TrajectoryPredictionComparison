import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# SETTINGS
# -------------------------
data_folder = "../data/final/train_filled"
val_folder = "../data/final/val_filled"

num_keypoints = 17
width, height = 640, 480
input_dim = 2 + num_keypoints * 2 + 1
past_len = 15
future_len = 60
epochs = 40
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# DATASET
# -------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, folder):
        self.X, self.Y = [], []

        for f in os.listdir(folder):
            if not f.endswith("_rgxb.json"):
                continue

            clip = self.load_clip(os.path.join(folder, f))

            for t in range(past_len, len(clip) - future_len):
                past = [clip[i]["feat"] for i in range(t - past_len, t)]
                future = [clip[i]["delta"] for i in range(t, t + future_len)]

                self.X.append(torch.tensor(past, dtype=torch.float32))
                self.Y.append(torch.tensor(future, dtype=torch.float32))

    def load_clip(self, json_path):
        with open(json_path) as f:
            data = json.load(f)

        clip = []
        prev_center = None
        for ann in data["annotations"]:
            # Skip frames without enough keypoints
            if len(ann.get("keypoints", [])) == 0 or len(ann["keypoints"][0]) < num_keypoints:
                # this should never trigger
                print("krise")
                continue
            else:
                kpts = np.array(ann["keypoints"][0])
                kpts[:,0] /= width
                kpts[:,1] /= height

            # Center between feet
            center = (kpts[15] + kpts[16]) / 2.0 if kpts.shape[0] >= 17 else prev_center

            # Delta from previous center
            delta = np.zeros(2) if prev_center is None else center - prev_center

            # Feature vector: center + keypoints flattened + depth from JSON
            feat = np.concatenate([center, kpts.flatten(), [ann.get("depth", 1.0)]])

            clip.append({"feat": feat, "delta": delta, "center": center})
            prev_center = center

        return clip

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = TrajectoryDataset(data_folder)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataset = TrajectoryDataset(val_folder)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------
# MODELS
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, future_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, future_len*2)
    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:,-1]
        out = self.fc(out)
        return out.view(-1, future_len, 2)

class GRUModel(nn.Module):
    def __init__(self, input_dim, future_len):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, future_len*2)
    def forward(self, x):
        out,_ = self.gru(x)
        out = out[:,-1]
        out = self.fc(out)
        return out.view(-1, future_len, 2)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, future_len):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 64)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(64, future_len*2)
    def forward(self, x):
        x = self.fc_in(x)
        attn,_ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn)
        x = x.transpose(1,2)
        x = self.pool(x).squeeze(-1)
        x = self.fc_out(x)
        return x.view(-1, future_len, 2)


# -------------------------
# CUSTOM ADE/FDE LOSS
# -------------------------
def ade_fde_loss(pred, target, alpha=1.0, eps=1e-6):
    """
    ADE/FDE loss for trajectory prediction.
    Adds a small epsilon to avoid zero gradients in edge cases.

    pred, target: tensors of shape (batch_size, future_len, 2)
    alpha: weight for FDE
    eps: small value to stabilize gradient computation
    """
    # Euclidean distance per frame
    dist = torch.norm(pred - target + eps, dim=-1)

    # ADE: mean over all future frames
    ade = dist.mean(dim=1)
    # FDE: distance at the final frame
    fde = dist[:, -1]

    # Combine and average over batch
    loss = ade.mean() + alpha * fde.mean()
    return loss


# -------------------------
# VALUDATION FUNCTION
# -------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = ade_fde_loss(pred, y, alpha=1.0)
            all_loss += loss.item() * x.size(0)
    return all_loss / len(dataloader.dataset)


# -------------------------
# TRAINING FUNCTION
# -------------------------
def train_model(model, val_dataloader=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = ade_fde_loss(pred, y, alpha=1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f}", end="")

        if val_dataloader is not None:
            val_loss = evaluate_model(model, val_dataloader, device)
            print(f" | Val Loss: {val_loss:.6f}")
        else:
            print()

    return model

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    models = {
        "lstm": LSTMModel(input_dim, future_len),
        "gru": GRUModel(input_dim, future_len),
        "transformer": TransformerModel(input_dim, future_len)
    }

    for name, model in models.items():
        print(f"\n=== TRAINING {name.upper()} ===")
        model = train_model(model, val_dataloader)
        torch.save(model, f"{name}_{past_len}_{future_len}.pt")
        print(f"Saved {name}_{past_len}_{future_len}.pt")