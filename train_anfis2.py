#!/usr/bin/env python3
"""
train_anfis2.py
Robust ANFIS trainer (3 inputs -> 1 output). Saves model state_dict and meta (mu,sigma,n_mfs).

Usage:
    python3 train_anfis2.py --train --data supervisor_log.csv --epochs 120 --n_mfs 5 --lr 0.001 --out anfis_v3.pth --meta anfis_v3_meta.json
"""

import argparse
import json
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import itertools

# -------------------------------------------------
# Gaussian Membership Function
# -------------------------------------------------
class GaussMf(nn.Module):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma, dtype=torch.float32)))

    def forward(self, x):
        sigma = torch.exp(self.log_sigma) + 1e-6
        return torch.exp(-0.5 * ((x - self.mean) / sigma) ** 2)


# -------------------------------------------------
# Corrected ANFIS: N-input, 1-output
# -------------------------------------------------
class ANFIS_Nin_1out(nn.Module):
    def __init__(self, n_inputs: int, n_mfs: int = 2):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_mfs = n_mfs

        # Total rules = inputs × MFs (NOT exponential)
        self.n_rules = n_inputs * n_mfs

        # -------------------------------------------------
        # Membership Functions per input
        # -------------------------------------------------
        self.mf_layers = nn.ModuleList()

        for _ in range(n_inputs):
            mfs = nn.ModuleList()
            centers = torch.linspace(-1.5, 1.5, n_mfs)
            for c in centers:
                mfs.append(GaussMf(c.item(), sigma=1.0))
            self.mf_layers.append(mfs)

        # -------------------------------------------------
        # Consequent parameters
        # One linear model per rule
        # y = a1*x1 + ... + aN*xN + b
        # -------------------------------------------------
        self.consequents = nn.Parameter(
            torch.randn(self.n_rules, n_inputs + 1) * 0.05
        )

    # -------------------------------------------------
    # Forward pass
    # -------------------------------------------------
    def forward(self, X):
        """
        X shape: [B, n_inputs]
        """
        B = X.shape[0]
        rule_activations = []

        # Compute MF outputs per input
        for i in range(self.n_inputs):
            xi = X[:, i]  # [B]
            for mf in self.mf_layers[i]:
                rule_activations.append(mf(xi))  # [B]

        # Stack → [B, n_rules]
        W = torch.stack(rule_activations, dim=1)

        # Normalize firing strengths
        W_norm = W / (W.sum(dim=1, keepdim=True) + 1e-6)

        # Consequent output
        X_ext = torch.cat([X, torch.ones(B, 1, device=X.device)], dim=1)
        Y_rules = torch.matmul(X_ext, self.consequents.T)

        # Weighted sum
        Y = torch.sum(W_norm * Y_rules, dim=1)
        return Y

# -----------------------------
# Data helpers
# -----------------------------
def load_csv(
    path: str,
    input_cols: tuple,
    target_col: str
) -> Tuple[np.ndarray, np.ndarray]:

    df = pd.read_csv(path)

    # Ensure all required columns exist
    required = list(input_cols) + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Convert to numeric safely
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required)

    X = df[list(input_cols)].values.astype(np.float32)
    Y = df[target_col].values.astype(np.float32)

    return X, Y


def normalize_fit_transform(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    Xn = (X - mu) / sigma
    return Xn, mu, sigma


# -----------------------------
# Training function
# -----------------------------
def train(
    X: np.ndarray,
    Y: np.ndarray,
    n_inputs: int,
    n_mfs: int = 3,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 0.01,
    device: str = "cpu"
):
    device = torch.device(device)

    # Normalize inputs
    Xn, mu, sigma = normalize_fit_transform(X)

    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    print(f"[DEBUG] Creating ANFIS with inputs={X.shape[1]}, mfs={n_mfs}, rules={n_mfs ** X.shape[1]}")

    # 🔴 IMPORTANT: use generic N-input ANFIS
    model = ANFIS_Nin_1out(
        n_inputs=n_inputs,
        n_mfs=n_mfs
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=15
    )

    loss_fn = nn.MSELoss()

    N = Xt.shape[0]
    print(f"Training samples: {N}, inputs: {n_inputs}, n_mfs: {n_mfs}")

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        model.train()
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx]
            yb = Yt[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= N
        scheduler.step(epoch_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss: {epoch_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.5f}"
            )

    return model.cpu(), mu.tolist(), sigma.tolist()
# -----------------------------
# Save helpers
# -----------------------------
def save_model_and_meta(
    model: nn.Module,
    mu: list,
    sigma: list,
    model_path: str,
    meta_path: str,
    n_mfs: int
):
    torch.save(model.state_dict(), model_path)

    meta = {
        "mu": mu,
        "sigma": sigma,
        "n_mfs": int(n_mfs)
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved model → {model_path}")
    print(f"[INFO] Saved meta  → {meta_path}")

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train multi-sensor ANFIS trust model")

    parser.add_argument("--train", action="store_true",
                        help="Enable training mode")
    parser.add_argument("--data", type=str, default="anfis_training_dataset2.csv",
                        help="CSV file containing training data")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--n_mfs", type=int, default=3,
                        help="Number of membership functions per input")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--out", type=str, default="anfis_multisensor.pth",
                        help="Output model file")
    parser.add_argument("--meta", type=str, default="anfis_multisensor_meta.json",
                        help="Output metadata file")

    args = parser.parse_args()

    if not args.train:
        parser.print_help()
        return

    # --------------------------------------------------
    # 1️⃣ Define input columns (MUST match CSV exactly)
    # --------------------------------------------------
    input_cols = (
        "sats",
        "hdop",
        "jump_m",
        "alt_err",
        "accel_vib",
        "gyro_mag"
    )
    target_col = "trust"

    # --------------------------------------------------
    # 2️⃣ Load dataset
    # --------------------------------------------------
    print(f"Loading training data from '{args.data}'...")
    X, Y = load_csv(
        args.data,
        input_cols=input_cols,
        target_col=target_col
    )

    # --------------------------------------------------
    # 3️⃣ Print diagnostics (CRITICAL for debugging)
    # --------------------------------------------------
    print("\n--- Data Diagnostics ---")
    for i, col in enumerate(input_cols):
        print(
            f"{col:12s} | "
            f"min: {X[:, i].min():8.4f} | "
            f"max: {X[:, i].max():8.4f} | "
            f"var: {X[:, i].var():10.6f}"
        )
    print(
        f"{target_col:12s} | "
        f"min: {Y.min():8.4f} | "
        f"max: {Y.max():8.4f} | "
        f"var: {Y.var():10.6f}"
    )
    print("------------------------\n")

    # --------------------------------------------------
    # 4️⃣ Train ANFIS model
    # --------------------------------------------------
    print("--- Starting ANFIS Training ---")
    model, mu, sigma = train(
        X,
        Y,
        n_inputs=len(input_cols),
        n_mfs=args.n_mfs,
        epochs=args.epochs,
        lr=args.lr,
        device="cpu"
    )
    print("--- Training Complete ---")

    # --------------------------------------------------
    # 5️⃣ Save model + normalization metadata
    # --------------------------------------------------
    save_model_and_meta(
        model=model,
        mu=mu,
        sigma=sigma,
        model_path=args.out,
        meta_path=args.meta,
        n_mfs=args.n_mfs
    )
if __name__ == "__main__":
    main()

