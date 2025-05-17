import torch
import torch.nn as nn
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")

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

def deploy_policy():
    model_path = os.path.join(MODEL_DIR, "bc_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    model = BCModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Example state to test the policy
    example_state = np.random.rand(1, 7)
    state_tensor = torch.tensor(example_state, dtype=torch.float32)
    predicted_action = model(state_tensor).detach().numpy()

    print("Input State: ", example_state)
    print("Predicted Action: ", predicted_action)

if __name__ == "__main__":
    try:
        deploy_policy()
    except FileNotFoundError as e:
        print(f"Deployment failed: {e}")
