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

DATA_FILE = os.path.join(DATA_DIR, "collected_data.hdf5")  # Updated to new dataset

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found at: {DATA_FILE}")

    with h5py.File(DATA_FILE, 'r') as f:
        demo_count = len(f) // 2
        if demo_count == 0:
            raise ValueError("No demonstration data found.")
        states = np.concatenate([f[f'states_{i}'][:] for i in range(demo_count)])
        actions = np.concatenate([f[f'actions_{i}'][:] for i in range(demo_count)])

    return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)

class BCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.net(x)

def train():
    try:
        states, actions = load_data()
    except (FileNotFoundError, ValueError) as e:
        print(f"Data loading failed: {e}")
        return

    model = BCModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(300):  # Slightly more epochs for better convergence
        preds = model(states)
        loss = loss_fn(preds, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model_path = os.path.join(MODEL_DIR, "bc_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved at: {model_path}")

if __name__ == "__main__":
    train()
