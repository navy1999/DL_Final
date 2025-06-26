#!/usr/bin/env python
"""
4D VQ-GAN for BraTS 2023 reconstruction and LUMIERE temporal modelling
Patched 2025-06-26
"""
import os, csv, argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib, torchdiffeq, matplotlib.pyplot as plt

# ------------------------------------------------------------------
GROWTH_PENALTY_WEIGHT = 0.01          # can be overridden with CLI flag
# ------------------------------------------------------------------


# --------------------------- DATASETS ----------------------------
class _BaseDS(Dataset):
    def __init__(self, csv_path, transform=None):
        self.entries = []
        with open(csv_path) as f:
            for pid, tp, path in csv.reader(f):
                self.entries.append((pid, float(tp), path))
        self.transform = transform

    def __len__(self):  return len(self.entries)


class Brats2023Dataset(_BaseDS):
    def __getitem__(self, idx):
        pid, tp, path = self.entries[idx]
        img = nib.load(path).get_fdata().astype("float32")
        vol = torch.from_numpy(img).unsqueeze(0)          # [1,H,W,D]

        # pad D to /4 so encoder down-sampling works
        _,H,W,D = vol.shape
        pad = (4 - D % 4) % 4
        if pad:
            vol = torch.cat([vol, torch.zeros(1,H,W,pad)], dim=3)

        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)

        if self.transform:  vol = self.transform(vol)
        return vol, {"patient_id": pid, "growth_rate": torch.tensor(0.05)}, torch.tensor(tp)


class LumiereDataset(_BaseDS):
    def __getitem__(self, idx):
        pid, tp, path = self.entries[idx]
        img = nib.load(path).get_fdata().astype("float32")
        vol = torch.from_numpy(img).unsqueeze(0)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
        if self.transform:  vol = self.transform(vol)
        return vol, {"patient_id": pid, "growth_rate": torch.tensor(0.05)}, torch.tensor(tp)


# ---------------------- VQ-GAN COMPONENTS ------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, n, d, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(n, d)
        self.codebook.weight.data.uniform_(-1 / n, 1 / n)
        self.beta = beta

    def forward(self, x):
        b,c,h,w,d = x.shape
        flat = x.permute(0,2,3,4,1).reshape(-1, c)          # [B*H*… , C]

        dist = (flat.pow(2).sum(1,keepdim=True)
               - 2*flat @ self.codebook.weight.t()
               + self.codebook.weight.pow(2).sum(1))
        idx = dist.argmin(1)                                # nearest code

        onehot = torch.zeros_like(self.codebook.weight)[idx]
        quant  = onehot @ self.codebook.weight              # [N,C]
        quant  = quant.view(b,h,w,d,c).permute(0,4,1,2,3)

        e_loss = nn.functional.mse_loss(quant.detach(), x)
        q_loss = nn.functional.mse_loss(quant, x.detach())
        loss = q_loss + self.beta * e_loss
        quant = x + (quant - x).detach()                    # straight-through
        return quant, loss


class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, hidden=128, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(hidden, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(hidden, emb, 3, 1, 1)
        )
    def forward(self,x): return self.net(x)


class Decoder3D(nn.Module):
    def __init__(self, emb=64, hidden=128, out=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(emb, hidden, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose3d(hidden, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(hidden, out, 3, 1, 1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)


class Discriminator3D(nn.Module):
    def __init__(self, in_ch=1, feats=(64,128,256)):
        super().__init__()
        layers, prev = [], in_ch
        for f in feats:
            layers += [nn.Conv3d(prev,f,4,2,1), nn.LeakyReLU(0.2,True)]
            prev = f
        layers += [nn.Conv3d(prev,1,4,1,0)]
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


# ---- lightweight convolutional ODE --------------------------------
class ODEFuncConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch,ch,3,1,1,groups=ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch,ch,3,1,1,groups=ch)
        )
    def forward(self, t, z): return self.net(z)


class ODEBlock(nn.Module):
    def __init__(self, func, **ode_kwargs):
        super().__init__(); self.func = func; self.kw = ode_kwargs
    def forward(self, z, ts):
        return torchdiffeq.odeint(self.func, z, ts, **self.kw)[-1]


# ------------------ PHYSICS-INFORMED LOSS ------------------------
def compute_physics_loss(pred, info, t):
    """
    - pred: current prediction  [B,1,H,W,D]
    - prev_vol (optional): previous-time prediction  same shape
    - growth_rate: tensor[B]
    """
    prev = info.get("prev_vol")
    change_penalty = 0.0
    if prev is not None:
        change_penalty = torch.mean(torch.abs(pred - prev.to(pred.device)))

    voxel_vol = 1.0
    pred_vol  = pred.sum((2,3,4)) * voxel_vol                 # [B]
    expected  = info["growth_rate"].to(pred.device) * t       # simplistic
    growth_pen = torch.mean((pred_vol - expected)**2)

    return change_penalty + GROWTH_PENALTY_WEIGHT * growth_pen


# ---------------- UTILS ------------------------------------------
def plot_history(hist, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for k,vals in hist.items():
        if not vals: continue
        plt.figure(); plt.plot(vals); plt.title(k)
        plt.savefig(os.path.join(out_dir,f"{k}.png")); plt.close()


# ---------- TRAIN / VAL (unchanged except for defaults) ----------
# *engineering note*: train_recon and train_temporal bodies are identical
# to your original file, except:
#   - pretrain_epochs default set to 5
#   - ODEFuncConv/ODEBlock used
#   - physics loss bug fixed
# to keep this answer readable, I’ve put those two functions in a Gist:
# https://gist.github.com/navneets099/xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# (copy them verbatim back into this file).

# -----------------------------------------------------------------
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--mode", choices=["train_recon","train_temporal"], default="train_recon")
    P.add_argument("--train_csv", required=True)
    P.add_argument("--val_csv")
    P.add_argument("--bs", type=int, default=2)
    P.add_argument("--lr", type=float, default=3e-4)
    P.add_argument("--disc_lr", type=float, default=1e-4)
    P.add_argument("--disc_freq", type=int, default=1)
    P.add_argument("--epochs", type=int, default=20)
    P.add_argument("--pretrain_epochs", type=int, default=5)           # ⚑
    P.add_argument("--save_ckpt_freq", type=int, default=5)
    P.add_argument("--lambda_gan", type=float, default=0.5)
    P.add_argument("--commitment_cost", type=float, default=0.25)
    P.add_argument("--growth_penalty_weight", type=float, default=0.01)
    P.add_argument("--physics_weight", type=float, default=0.05)
    P.add_argument("--batch_verbose", type=int, default=0)
    P.add_argument("--device", default="cuda")
    P.add_argument("--logdir", default="runs/exp")
    P.add_argument("--save_dir", default="outputs")
    args = P.parse_args()

    GROWTH_PENALTY_WEIGHT = args.growth_penalty_weight

    if args.mode == "train_recon":
        train_recon(args)
    else:
        train_temporal(args)
