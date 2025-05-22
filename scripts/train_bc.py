import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) #
DATA_DIR = os.path.join(BASE_DIR, "data") #
MODEL_DIR = os.path.join(BASE_DIR, "models") #

os.makedirs(MODEL_DIR, exist_ok=True) #

DATA_FILE = os.path.join(DATA_DIR, "ik_demos.hdf5") #

def load_data():
    if not os.path.exists(DATA_FILE): #
        raise FileNotFoundError(f"Data file not found at: {DATA_FILE}") #

    all_states = []
    all_actions = []
    with h5py.File(DATA_FILE, 'r') as f: #
        # More robust way to find all demo data
        state_keys = sorted([key for key in f.keys() if key.startswith('states_')]) #
        action_keys = sorted([key for key in f.keys() if key.startswith('actions_')]) #
        
        num_demos_in_file = len(state_keys) #
        print(f"Found {num_demos_in_file} demonstrations in HDF5 file.")

        for state_key in state_keys: #
            action_key = state_key.replace('states_', 'actions_') #
            if action_key in f: #
                all_states.append(f[state_key][:]) #
                all_actions.append(f[action_key][:]) #
            else:
                print(f"Warning: Missing corresponding action key for {state_key}. Skipping.")
    
    if not all_states or not all_actions: #
        raise ValueError("No data loaded. Check HDF5 file content and keys.") #

    states_np = np.concatenate(all_states, axis=0) #
    actions_np = np.concatenate(all_actions, axis=0) #
    
    print(f"Loaded data shapes: States {states_np.shape}, Actions {actions_np.shape}")


    return torch.tensor(states_np, dtype=torch.float32), torch.tensor(actions_np, dtype=torch.float32) #

# Using the model definition from your uploaded train_bc.py
class AdvancedBCModel(nn.Module): #
    def __init__(self): #
        super().__init__() #
        self.net = nn.Sequential( #
            nn.Linear(10, 512), #
            nn.BatchNorm1d(512), #
            nn.GELU(), #
            nn.Dropout(0.3), #
            nn.Linear(512, 512), #
            nn.BatchNorm1d(512), #
            nn.GELU(), #
            nn.Dropout(0.3), #
            nn.Linear(512, 256), #
            nn.BatchNorm1d(256), #
            nn.GELU(), #
            nn.Linear(256, 128), #
            nn.GELU(), #
            nn.Linear(128, 7)  # Output layer for 7 joint positions #
        )

    def forward(self, x): #
        return self.net(x) #

def train():
    states, actions = load_data() #

    if states.shape[0] == 0 or actions.shape[0] == 0:
        print("Error: No data loaded for training. Exiting.")
        return
    
    if states.shape[0] != actions.shape[0]:
        print(f"Error: Mismatch in number of states ({states.shape[0]}) and actions ({actions.shape[0]}). Exiting.")
        return

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    print(f"Training with State Dimension: {state_dim}, Action Dimension: {action_dim}")


    model = AdvancedBCModel() # # Your model definition
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) #
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50) # # verbose removed
    loss_fn = nn.MSELoss() #

    num_epochs = 2000 #
    batch_size = 256  # Recommended for batching
    
    dataset = torch.utils.data.TensorDataset(states, actions) #
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) #

    for epoch in range(num_epochs): #
        model.train() #
        epoch_loss = 0.0
        num_batches = 0
        # --- Batched Training Loop ---
        for batch_states, batch_actions in dataloader: #
            preds = model(batch_states) #
            loss = loss_fn(preds, batch_actions) #

            optimizer.zero_grad() #
            loss.backward() #
            optimizer.step() #
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        scheduler.step(avg_epoch_loss) # Pass average epoch loss to scheduler #

        if (epoch + 1) % 100 == 0: #
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}") #

    model_path = os.path.join(MODEL_DIR, "bc_model.pth") #
    torch.save(model.state_dict(), model_path) #
    print(f"Training complete. Model saved at: {model_path}") #

if __name__ == "__main__":
    train() #