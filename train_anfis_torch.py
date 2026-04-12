# train_anfis_torch.py
"""
Self-contained ANFIS trainer in PyTorch.

Usage:
  - Put your labeled CSV 'final_labeled_dataset.csv' in same folder.
  - CSV must contain columns: satellites, hdop, vibration_x, target_trust_score
  - Run: python3 train_anfis_torch.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
from itertools import product
from typing import List

# ----------------- Configuration -----------------
DATA_FILE = "labeled_dataset.csv"
MODEL_OUTPUT_FILE = "anfis_model_torch.pth"
EPOCHS = 200
LR = 1e-2
BATCH_SIZE = 64
N_MFS_PER_INPUT = 3   # number of membership functions per input
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------- Data helpers -------------------
def load_data(filename):
    df = pd.read_csv(filename)
    input_cols = ['satellites', 'hdop', 'vibration_x']
    target_col = 'target_trust_score'
    if not set(input_cols + [target_col]).issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {input_cols + [target_col]}")
    X = df[input_cols].values.astype(np.float32)
    Y = df[[target_col]].values.astype(np.float32)
    return X, Y

def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
    if std is None:
        std = X.std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mean) / std
    return Xs, mean, std

# ----------------- ANFIS modules ------------------
class GaussianMF(nn.Module):
    """Gaussian MF with center c and width sigma (both trainable)."""
    def __init__(self, init_center, init_sigma):
        super().__init__()
        # store as parameters
        self.c = nn.Parameter(torch.tensor(float(init_center)))
        # parameterize sigma via softplus to ensure >0
        self._sigma_param = nn.Parameter(torch.tensor(float(init_sigma)))
    def forward(self, x):
        # x shape: (batch,)
        sigma = torch.nn.functional.softplus(self._sigma_param) + 1e-6
        # gaussian: exp(-0.5 * ((x-c)/sigma)^2)
        return torch.exp(-0.5 * ((x - self.c) / sigma) ** 2)

class ANFISNet(nn.Module):
    def __init__(self, n_inputs: int, n_mfs_per_input: int, input_centers: List[List[float]] = None):
        """
        n_inputs: number of input features (3)
        n_mfs_per_input: e.g. 3
        input_centers: optional initial centers, list of list, length n_inputs x n_mfs
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs_per_input
        # Membership function objects stored per input
        self.mfs = nn.ModuleList()
        for i in range(n_inputs):
            centers_i = None
            if input_centers is not None:
                centers_i = input_centers[i]
            # create n_mfs GaussianMF for this input
            mfs_for_input = nn.ModuleList()
            for j in range(n_mfs_per_input):
                if centers_i is not None:
                    c = centers_i[j]
                else:
                    c = (j - (n_mfs_per_input-1)/2.0)  # default spacing; will be fine with standardized inputs
                sigma_init = 1.0
                mfs_for_input.append(GaussianMF(c, sigma_init))
            self.mfs.append(mfs_for_input)

        # build rule-table: each rule is a combination of mf indices for each input
        # number of rules = n_mfs^n_inputs
        self.rule_combinations = list(product(range(self.n_mfs), repeat=self.n_inputs))
        self.n_rules = len(self.rule_combinations)

        # consequent parameters: for each rule, linear coefficients for inputs + bias
        # we'll parametrize as (n_rules x n_inputs) and (n_rules x 1)
        self.consequent_lin = nn.Parameter(torch.randn(self.n_rules, self.n_inputs) * 0.1)
        self.consequent_bias = nn.Parameter(torch.zeros(self.n_rules, 1))

    def forward(self, x):
        """
        x: (batch, n_inputs)
        returns: (batch, 1)
        """
        batch = x.shape[0]
        # compute degrees: for each input i, compute (batch, n_mfs)
        degrees_per_input = []
        for i in range(self.n_inputs):
            xi = x[:, i]  # (batch,)
            # compute each mf output
            mfi_outputs = []
            for mf in self.mfs[i]:
                mfi_outputs.append(mf(xi))  # (batch,)
            # stack -> (batch, n_mfs)
            degrees_per_input.append(torch.stack(mfi_outputs, dim=1))

        # compute firing strengths for each rule
        # for rule r, pick for each input i the degree at index rule_combinations[r][i], then multiply across inputs
        # we can gather efficiently:
        # build index tensor of shape (n_rules, n_inputs)
        idx = torch.tensor(self.rule_combinations, device=x.device, dtype=torch.long)  # (n_rules, n_inputs)

        # for each input i: degrees_per_input[i] is (batch, n_mfs)
        # select for all rules: degrees_per_input[i][:, idx[:,i]] -> (batch, n_rules)
        firing = torch.ones((batch, self.n_rules), device=x.device)
        for i in range(self.n_inputs):
            dj = degrees_per_input[i]  # (batch, n_mfs)
            sel = dj[:, idx[:, i]]     # (batch, n_rules)
            firing = firing * (sel + 1e-9)

        # normalized firing strengths
        firing_sum = firing.sum(dim=1, keepdim=True) + 1e-9
        firing_norm = firing / firing_sum  # (batch, n_rules)

        # compute consequent outputs: for each rule r, y_r = a_r . x + b_r
        # a: (n_rules, n_inputs), x: (batch, n_inputs) -> we want (batch, n_rules)
        # compute x @ a.T -> (batch, n_rules)
        cons_lin = torch.matmul(x, self.consequent_lin.t()) + self.consequent_bias.t()  # (batch, n_rules)
        # final output is sum_r (firing_norm[:,r] * cons_lin[:,r])
        y = (firing_norm * cons_lin).sum(dim=1, keepdim=True)  # (batch,1)
        return y

# ---------------- Training & Utilities ----------------
def train(model, X_train, Y_train, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    model.to(DEVICE)
    X = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(Y_train, dtype=torch.float32, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")
    return model

# ---------------- Main ------------------------------
def main():
    print("Loading data...")
    X_raw, Y_raw = load_data(DATA_FILE)
    # standardize features
    Xs, mean, std = standardize(X_raw)
    # scale targets to [0,1] if not already (assume trust score is already 0..1, but safe guard)
    Y = Y_raw
    if Y.max() > 1.001 or Y.min() < -0.001:
        # scale to 0..1
        Y = (Y - Y.min()) / (Y.max() - Y.min() + 1e-9)

    print(f"Data shapes: X={Xs.shape}, Y={Y.shape}")
    n_inputs = Xs.shape[1]

    # initialize centers per input: spread across -1..1 in standardized space
    input_centers = []
    for i in range(n_inputs):
        # choose centers evenly spaced between -1.5 and 1.5
        centers = np.linspace(-1.5, 1.5, N_MFS_PER_INPUT).tolist()
        input_centers.append(centers)

    model = ANFISNet(n_inputs=n_inputs, n_mfs_per_input=N_MFS_PER_INPUT, input_centers=input_centers)
    print("Starting training on device:", DEVICE)
    trained = train(model, Xs, Y, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE)

    # save model + normalization params
    state = {
        "model_state_dict": trained.state_dict(),
        "mean": mean,
        "std": std,
        "n_inputs": n_inputs,
        "n_mfs": N_MFS_PER_INPUT,
        "input_centers": input_centers
    }
    torch.save(state, MODEL_OUTPUT_FILE)
    print(f"Saved trained ANFIS to {MODEL_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
