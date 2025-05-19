
# 🤖 7-DOF Robotic Arm Pick-and-Place with Imitation Learning

A complete end-to-end project using a 7-DOF Franka Emika Panda robotic arm in Mujoco simulation, trained with Behavioral Cloning (Imitation Learning) to perform a pick-and-place task.

---

## 📚 Project Overview

This project demonstrates:
- ✅ Simulation of a 7-DOF robotic arm using Mujoco.
- ✅ Scripted demonstration data collection using Cartesian control and mocap.
- ✅ Training a deep Behavioral Cloning (BC) policy using a highly complex neural network.
- ✅ Intelligent simulation loop with dynamic gripper control and task success verification.

---

## 🚀 Project Structure

```
7DOF_Robotic_Manipulation_IL/
├── data/                  # Collected demonstration datasets
├── models/                # Trained BC models and XML files
├── scripts/               # All project scripts
│   ├── ik_data_collector.py
│   ├── train_bc.py
│   └── simulation_loop.py
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## 📦 Setup Instructions

### 1. Install Dependencies
```bash
pip install mujoco mujoco-py torch numpy h5py matplotlib
```
Ensure Mujoco is installed and properly configured. You can download Mujoco from [Mujoco Website](https://mujoco.org/).

---

### 2. Collect Demonstration Data
```bash
python scripts/ik_data_collector.py
```
- Uses mocap control to generate realistic Cartesian demonstrations.
- Stores data in `data/ik_demos.hdf5`.

---

### 3. Train the Behavioral Cloning Model
```bash
python scripts/train_bc.py
```
- Uses an advanced neural network with BatchNorm, Dropout, GELU activations, and adaptive learning rate scheduling.
- Trains for 2000 epochs and saves the model to `models/bc_model.pth`.

---

### 4. Run the Simulation
```bash
python scripts/simulation_loop.py
```
- Loads the trained model and controls the arm in simulation.
- Dynamically controls the gripper based on proximity to the object.
- Prints task success metrics at the end.

---

## 🧩 Key Features
- ✔️ Advanced Policy Network with 512 hidden units and deep architecture.
- ✔️ Hysteresis-based intelligent gripper control.
- ✔️ Simulation success tracking based on object movement.
- ✔️ Phase-based data collection using mocap for realistic demonstrations.

---

## 📊 Results
- The final policy successfully moves the arm toward the object and attempts to pick it up.
- Success rate improves significantly after integrating complex model architecture and high-quality demonstrations.

---

## 📅 Project Timeline
| Phase             | Estimated Time |
|-------------------|----------------|
| Environment Setup | 1 Day           |
| Data Collection   | 2 Days          |
| Model Training    | 1 Day (Including Hyperparameter Tuning) |
| Simulation & Testing | 1 Day        |

---

## 📖 Future Improvements
- Integrate Vision-Based Inputs using simulated cameras.
- Fine-tune the policy using Reinforcement Learning after BC (Hybrid BC+RL).
- Implement Sim2Real Transfer for real-world deployment.

---

✨ **Created by [Your Name]**
