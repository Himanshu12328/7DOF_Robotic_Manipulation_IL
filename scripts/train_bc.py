import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "ik_demos.hdf5")

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found at: {DATA_FILE}")

    with h5py.File(DATA_FILE, 'r') as f:
        demo_count = len(f) // 2
        states = np.concatenate([f[f'states_{i}'][:] for i in range(demo_count)])
        actions = np.concatenate([f[f'actions_{i}'][:] for i in range(demo_count)])

    return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)

class AdvancedBCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 7)  # Output layer for 7 joint positions
        )

    def forward(self, x):
        return self.net(x)

def train():
    states, actions = load_data()

    model = AdvancedBCModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    loss_fn = nn.MSELoss()

    for epoch in range(2000):
        model.train()
        preds = model(states)
        loss = loss_fn(preds, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    model_path = os.path.join(MODEL_DIR, "bc_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved at: {model_path}")

if __name__ == "__main__":
    train()
