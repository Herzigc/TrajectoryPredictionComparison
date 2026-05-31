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
# center + velocity + acceleration + keypoints + depth
input_dim = 2 + 2 + 2 + num_keypoints * 2 + 1
past_len = 30
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
                future = [clip[i]["velocity"] for i in range(t, t + future_len)]

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

            velocity = np.zeros(2) if prev_center is None else center - prev_center

            acceleration = np.zeros(2)
            if len(clip) > 0:
                prev_velocity = clip[-1]["velocity"]
                acceleration = velocity - prev_velocity

            feat = np.concatenate([
                center,
                velocity,
                acceleration,
                kpts.flatten(),
                [ann.get("depth", 1.0)]
            ])

            clip.append({
                "feat": feat,
                "velocity": velocity,
                "center": center
            })

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
# CUSTOM ADE/FDE LOSS
# -------------------------
def ade_fde_loss(pred, target, alpha=1.0):
    pred_traj = torch.cumsum(pred, dim=1)
    target_traj = torch.cumsum(target, dim=1)

    dist = torch.norm(pred_traj - target_traj, dim=-1)

    ade = dist.mean(dim=1)
    fde = dist[:, -1]

    return ade.mean() + alpha * fde.mean()


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
        torch.save(model, f"{name}_{past_len}_{future_len}_new.pt")
        print(f"Saved {name}_{past_len}_{future_len}_new.pt")