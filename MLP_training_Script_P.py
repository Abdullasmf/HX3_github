import scipy.io
import numpy as np
from pathlib import Path
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# Weight/Power training script (tuned for reduced overfitting).
# Changes:
#  - Smaller network (hidden_size 512, layers 4)
#  - Dropout 0.1
#  - Weight decay 1e-3
#  - Patience 400, max epochs 30000
#  - Adaptive batch size target 256
#  - Captures val_r2_history
hidden_size = 1024*1  # revert baseline size
layers = 2
num_epochs = 30000
target_batch_size = 2048  # original larger batch

model_name = f"P_MLP_{hidden_size}_{layers}"
print(f"Model name: {model_name}")
from models import  get_device, TaperedMultiLayerNN
current_dir = project_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
save_path = Path(current_dir, f"{model_name}.pth")
data_path = Path(current_dir , "HX3_LHS_finalData_uniformlhs.mat")
device = get_device()
data = scipy.io.loadmat(data_path)
variable_data = data['HXdata']
print(f"Data shape: {variable_data.shape}")
variable_data = variable_data[:, 2:]#


power_too_high = variable_data[:,15] >= 15000
weight_too_low = variable_data[:,14] <= 0
drag_too_low = variable_data[:,16] < -1500

variable_data = variable_data[~power_too_high & ~weight_too_low & ~drag_too_low]
print(f"removed {np.sum(power_too_high | weight_too_low | drag_too_low)} rows based on weight/power/drag criteria")

#only include fan-on cases
fan_off = variable_data[:, 13] == 0
variable_data = variable_data[~fan_off]
print(f"num of fan-off rows removed (keeping fan-on): {np.sum(fan_off)}")
print(variable_data[:,13])
no_div = variable_data[:, 27] == 0
variable_data = variable_data[~no_div]
print(f"num of no-div rows removed: {np.sum(no_div)}")


col_names = [
    "coolant channel diameter (m)",         # 0
    "HX overall length (m)",                # 1
    "HX overall width (m)",                 # 2
    "Channel height (m)",                   # 3
    "Number of air layers/channels",        # 4
    "Strut Diameter (m)",                   # 5
    "Strut length to diameter ratio",       # 6
    "coolant flow rate (kg/s)",             # 7
    "Air flow rate (kg/s)",                 # 8
    "Area ratio_diff (diffuser exit/inlet)",# 9
    "diffuser half angle (degrees)",        # 10
    "Area ratio_nozz (nozzle inlet/exit)",  # 11
    "nozzle half angle (degrees)",          # 12
    "fan on/off",                           # 13
    "HX weight (kg)",                       # 14
    "HX power (W)",                         # 15
    "Drag (N)",                             # 16
    "Design point (780=Takeoff, 2720=Cruise)", # 17
    "DTAMB (centigrade)",                   # 18
    "T_Bat_in or T_HX_out (centigrade)",    # 19
    "T_HX_in limit (centigrade)",           # 20
    "Heat Load (W)",                        # 21
    "Actual T_HX_in (centigrade)",          # 22
    "porosity",                             # 23
    "air side hydraulic diameter (wrt DP)", # 24
    "HX overall height (m)",                # 25
    "aircraft speed"                        # 26
]


y_indicies = [15, 21]

remove_cols = [14, 15, 16, 21, 13, 27, 19,20,17,25,24,23]
x = np.delete(variable_data, remove_cols, axis=1)
y = variable_data[:, y_indicies]
# Add ALL engineered features to x (original + new)
# x = np.column_stack((x, AS_hyd_dia, HX_front_area, additional_features))





print("Input features shape:", x.shape)
print("Output targets shape:", y.shape)


# Split BEFORE fitting scalers to avoid leakage
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    x, y, test_size=0.20, random_state=42, shuffle=True
)

########################################
# Fit scalers only on training data
########################################
input_scaler = StandardScaler().fit(X_train_raw)
output_scaler = StandardScaler().fit(y_train_raw)  # multi-output standardization

########################################
# (No sampler / weighting: plain baseline)
########################################

# Transform to scaled tensors
X_train = torch.tensor(input_scaler.transform(X_train_raw), dtype=torch.float32).to(device)
X_val   = torch.tensor(input_scaler.transform(X_val_raw), dtype=torch.float32).to(device)
y_train = torch.tensor(output_scaler.transform(y_train_raw), dtype=torch.float32).to(device)
y_val   = torch.tensor(output_scaler.transform(y_val_raw), dtype=torch.float32).to(device)

print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Validation set: X_val {X_val.shape}, y_val {y_val.shape}")

# For compatibility with subsequent code blocks referencing these names
X_tensor = torch.cat([X_train, X_val], dim=0)
y_tensor = torch.cat([y_train, y_val], dim=0)


# (Split already performed prior to scaling to prevent leakage)

# Optional: Create DataLoaders for batch training


########################################
# Create datasets
########################################
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Simple DataLoaders
batch_size = min(target_batch_size, len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"Using batch size: {batch_size}")

# Initialize the model
input_size = X_train.shape[1]  # Number of input features
output_size = y_train.shape[1]  # Number of output targets

model = TaperedMultiLayerNN(
    input_size=input_size,
    initial_hidden_size=hidden_size,
    output_size=output_size,
    num_layers=layers,
    dropout=0.10
).to(device)

print("Model architecture:")
print(f"Input size: {input_size}")
print(f"Hidden layers: {hidden_size}")
print(f"Output size: {output_size}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Loss & optimizer (plain MSE baseline)
criterion = nn.MSELoss()
base_lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40, verbose=True, min_lr=1e-6)
print(f"Initialized baseline (MSE) optimizer: lr={base_lr}, wd=1e-3")

# -------------------------
# Training configuration extras
# -------------------------
early_stop_patience = 400    # epochs with no val loss improvement before stopping
improvement_min_delta = 1e-7 # minimum decrease in val loss to count as improvement
log_interval = 100           # print losses every N epochs
clip_grad_norm = 1.0         # set None/0 to disable gradient clipping
compute_r2_every = 100       # compute and log R2 every N epochs (can be same as log_interval)

def compute_r2_and_baseline(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Return per-output R2 and baseline (mean) MSE in the current (scaled) space.
    R2 is invariant to later inverse scaling; computing here is fine.
    """
    # y_true, y_pred: [N, O]
    with torch.no_grad():
        mean = y_true.mean(dim=0, keepdim=True)
        ss_res = torch.sum((y_true - y_pred)**2, dim=0)
        ss_tot = torch.sum((y_true - mean)**2, dim=0)
        # Avoid division by zero (if variance zero -> R2 undefined => set to 0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        baseline_mse = torch.mean((y_true - mean)**2, dim=0)
    return r2, baseline_mse

def format_tensor_1d(t: torch.Tensor):
    return '[' + ', '.join(f"{v:.4f}" for v in t.tolist()) + ']'

#############################
# Inline Training Loop
#############################
train_losses = []
val_losses = []
val_r2_history = []
best_val_loss = float('inf')
best_val_r2 = -1e9
epochs_no_improve = 0

torch.manual_seed(42); np.random.seed(42)
print("Starting training (baseline MSE first)...")
for epoch in range(num_epochs):
    model.train()
    run_tr = 0.0
    for bx, by in train_loader:
        bx = bx.to(device)
        by = by.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        if clip_grad_norm and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        run_tr += loss.item()

    model.eval()
    run_val = 0.0
    all_p = []
    all_t = []
    with torch.no_grad():
        # validation pass
        for bx, by in val_loader:
            bx = bx.to(device)
            by = by.to(device)
            pred = model(bx)
            vloss = criterion(pred, by)
            run_val += vloss.item()
            if (epoch + 1) % compute_r2_every == 0:
                all_p.append(pred)
                all_t.append(by)

    avg_tr = run_tr / max(1, len(train_loader))
    avg_va = run_val / max(1, len(val_loader))
    train_losses.append(avg_tr)
    val_losses.append(avg_va)

    mean_r2 = None
    if (epoch + 1) % compute_r2_every == 0 and all_p:
        cat_p = torch.cat(all_p, dim=0)
        cat_t = torch.cat(all_t, dim=0)
        r2_vec, _ = compute_r2_and_baseline(cat_t, cat_p)
        mean_r2 = r2_vec.mean().item()
        val_r2_history.append(mean_r2)

    improved_loss = (best_val_loss - avg_va) > improvement_min_delta
    improved_r2 = mean_r2 is not None and (mean_r2 - best_val_r2) > 1e-4
    if improved_loss or improved_r2:
        if improved_loss:
            best_val_loss = avg_va
        if improved_r2:
            best_val_r2 = mean_r2
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_r2': best_val_r2,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_history': val_r2_history,
            'input_scaler': input_scaler,
            'output_scaler': output_scaler,
            'input_cols': [col_names[i] for i in range(len(col_names)) if i not in remove_cols],
            'output_cols': [col_names[i] for i in y_indicies],
            'config': {
                'hidden_size': hidden_size,
                'layers': layers,
                'loss': 'MSE',
                'batch_size': batch_size,
            }
        }, save_path)
        if (epoch + 1) % log_interval == 0:
            if mean_r2 is not None:
                print(f"[Epoch {epoch+1}] ValLoss {avg_va:.6e} MeanR2 {mean_r2:.4f} LR {optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"[Epoch {epoch+1}] ValLoss {avg_va:.6e} LR {optimizer.param_groups[0]['lr']:.2e}")
    else:
        epochs_no_improve += 1
        if (epoch + 1) % log_interval == 0:
            if mean_r2 is not None:
                print(f"Epoch {epoch+1}: Train {avg_tr:.6e} | Val {avg_va:.6e} | R2 {mean_r2:.4f} | no improve {epochs_no_improve} LR {optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"Epoch {epoch+1}: Train {avg_tr:.6e} | Val {avg_va:.6e} | no improve {epochs_no_improve} LR {optimizer.param_groups[0]['lr']:.2e}")

    scheduler.step(avg_va)
    if epochs_no_improve >= early_stop_patience:
        print(f"Early stopping at epoch {epoch+1}. Best loss {best_val_loss:.6e} best R2 {best_val_r2:.4f}")
        break

print(f'Training completed! Best validation loss: {best_val_loss:.6f} | Best R2: {best_val_r2:.4f}')

print("Training completed! R2 history entries:", len(val_r2_history))