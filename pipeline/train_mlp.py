import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from .config import BATCH_SIZE, EPOCHS, LAYER_SIZES, LEARNING_RATE, MODEL_DIR, PATIENCE, SEED
from .models import DualBranchMLP


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


@torch.no_grad()
def predict_numpy(model, X, device, batch=2048):
    model.eval()
    outs = []
    x = torch.as_tensor(X, dtype=torch.float32)
    for i in range(0, len(x), batch):
        outs.append(model(x[i : i + batch].to(device)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_one_target(name, X_train, y_train, X_test, y_test, device):
    set_seed()
    n_features = X_train.shape[1]
    model = DualBranchMLP(n_features, LAYER_SIZES[name]).to(device)

    # Standardize target for numerical stability during training
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-6:
        y_std = 1.0

    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    loss_fn = torch.nn.MSELoss()

    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train_norm, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.as_tensor(X_test, dtype=torch.float32),
        torch.as_tensor(y_test_norm, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    history = {"epoch": [], "train_loss": [], "test_loss": []}
    best_state = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            running += loss.item() * len(xb)
            n += len(xb)
        train_loss = running / max(n, 1)

        model.eval()
        running = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                running += loss.item() * len(xb)
                n += len(xb)
        test_loss = running / max(n, 1)
        scheduler.step()

        # Record physical-scale MSE loss for clear interpretability
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss * (y_std ** 2))
        history["test_loss"].append(test_loss * (y_std ** 2))
        print(f"  {name} epoch {epoch}/{EPOCHS}  train_mse={train_loss * (y_std ** 2):.2f}  test_mse={test_loss * (y_std ** 2):.2f}")

        if test_loss + 1e-6 < best_val:
            best_val = test_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  early stop {name} at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODEL_DIR / f"{name}_best_network.pth"
    torch.save(model.state_dict(), ckpt)

    # Invert scaling to obtain predictions on original physical scale
    y_tr = predict_numpy(model, X_train, device) * y_std + y_mean
    y_te = predict_numpy(model, X_test, device) * y_std + y_mean

    # Wrap model for forward prediction on physical scale
    class ScaledModel(torch.nn.Module):
        def __init__(self, base_m, m, s):
            super().__init__()
            self.base_m = base_m
            self.m = m
            self.s = s
        def forward(self, x):
            return self.base_m(x) * self.s + self.m

    wrapped_model = ScaledModel(model, y_mean, y_std)

    return {
        "model": wrapped_model,
        "history": history,
        "train": _metrics(y_train, y_tr),
        "test": _metrics(y_test, y_te),
        "y_train_true": y_train,
        "y_train_pred": y_tr,
        "y_test_true": y_test,
        "y_test_pred": y_te,
        "ckpt": str(ckpt),
    }


def save_scaler(scaler, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
