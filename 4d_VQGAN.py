"""
4D VQ-GAN for Longitudinal Brain Tumor MRI Volumes

This script provides:
 1. LongitudinalMRI Dataset loader
 2. 3D VQ-GAN components (Encoder, Decoder, VectorQuantizer)
 3. Neural ODE temporal modeling (ODEFunc, ODEBlock)
 4. Physics-informed loss with biomechanical constraints
 5. 3D Discriminator (PatchGAN style)
 6. Adversarial training (VQ-GAN)
 7. Training loops: basic & temporal + physics + GAN
 8. TensorBoard logging, validation, and loss-curve plotting

Dataset Download & Preparation:
 1. **Download BraTS** (e.g., 2021) from the MICCAI site or Kaggle:
      https://www.med.upenn.edu/cbica/brats2021/
 2. Extract to a folder (`BraTS21`) so each patient has a subfolder containing
     timepoint scans: e.g. `BraTS21/Patient001/scan_t0.nii.gz`, `scan_t1.nii.gz`, etc.
 3. Create CSVs (`train.csv`, `val.csv`) with rows:
      ```
      Patient001,0,/path/to/BraTS21/Patient001/scan_t0.nii.gz
      Patient001,1,/path/to/BraTS21/Patient001/scan_t1.nii.gz
      ```

Usage:
 ```bash
 pip install torch torchvision torchdiffeq nibabel tensorboard matplotlib
 python fourd_vqgan.py --mode train_temporal \
                       --train_csv train.csv \
                       --val_csv val.csv \
                       --logdir runs/exp1 \
                       --save_dir checkpoints
 ```

Options:
  --mode {train,train_temporal}
  --train_csv PATH      : training CSV file
  --val_csv PATH        : validation CSV file (optional)
  --bs INT              : batch size (default:2)
  --lr FLOAT            : learning rate (default:1e-3)
  --epochs INT          : number of epochs (default:10)
  --device STR          : device ("cuda" or "cpu")
  --logdir STR          : TensorBoard logs dir (default:"runs/exp")
  --save_dir STR        : directory to save plots/checkpoints (default:"outputs")
"""
import os
import argparse
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import torchdiffeq
import matplotlib.pyplot as plt

# ---------------------- Dataset Loader ----------------------
class LongitudinalMRIDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.entries = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for pid, tp, path in reader:
                self.entries.append((pid, float(tp), path))
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, timepoint, path = self.entries[idx]
        img = nib.load(path).get_fdata().astype('float32')
        vol = torch.from_numpy(img).unsqueeze(0)  # [1,H,W,D]
        if self.transform:
            vol = self.transform(vol)
        return vol, {'patient_id': pid, 'growth_rate': 0.05}, torch.tensor(timepoint)

# ---------------------- VQ-GAN Components ----------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        B,C,H,W,D = x.shape
        flat = x.permute(0,2,3,4,1).reshape(-1, C)
        dist = (flat.pow(2).sum(1,keepdim=True)
               - 2*flat @ self.embedding.weight.t()
               + self.embedding.weight.pow(2).sum(1))
        idx = torch.argmin(dist,dim=1)
        enc = torch.zeros(idx.size(0), self.embedding.num_embeddings, device=x.device)
        enc.scatter_(1, idx.unsqueeze(1), 1)
        quant_flat = enc @ self.embedding.weight
        quant = quant_flat.view(B,H,W,D,C).permute(0,4,1,2,3)
        e_loss = nn.functional.mse_loss(quant.detach(), x)
        q_loss = nn.functional.mse_loss(quant, x.detach())
        loss = q_loss + self.commitment_cost * e_loss
        quant = x + (quant - x).detach()
        return quant, loss

class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, hidden=128, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch,hidden,4,2,1), nn.ReLU(),
            nn.Conv3d(hidden,hidden,4,2,1), nn.ReLU(),
            nn.Conv3d(hidden,emb,3,1,1)
        )
    def forward(self,x): return self.net(x)

class Decoder3D(nn.Module):
    def __init__(self, emb=64, hidden=128, out=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(emb,hidden,4,2,1), nn.ReLU(),
            nn.ConvTranspose3d(hidden,hidden,4,2,1), nn.ReLU(),
            nn.Conv3d(hidden,out,3,1,1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)

# ---------------------- Discriminator ----------------------
class Discriminator3D(nn.Module):
    def __init__(self, in_ch=1, features=[64,128,256]):
        super().__init__()
        layers = []
        prev = in_ch
        for f in features:
            layers += [nn.Conv3d(prev, f, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            prev = f
        layers += [nn.Conv3d(prev, 1, 4, 1, 0)]  # PatchGAN
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------- Neural ODE Temporal ----------------------
class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,dim*2), nn.ReLU(), nn.Linear(dim*2,dim)
        )
    def forward(self,t,z): return self.net(z)

class ODEBlock(nn.Module):
    def __init__(self, func, method='rk4', tol=1e-3):
        super().__init__()
        self.func, self.method, self.tol = func, method, tol
    def forward(self,z,timepts):
        B,C,H,W,D = z.shape
        flat = z.reshape(B, -1)
        out = torchdiffeq.odeint(self.func, flat, timepts.to(flat.device),
                                 method=self.method, atol=self.tol, rtol=self.tol)
        return out[-1].reshape(B,C,H,W,D)

# ---------------------- Physics-Informed Loss ----------------------
def compute_physics_loss(pred, info, t):
    # 1. Smoothness
    diff = pred - pred.detach()
    change_penalty = torch.mean(torch.abs(diff) / (t.view(-1,1,1,1,1) + 1e-6))
    # 2. Volume growth consistency
    voxel_vol = 1.0
    pred_vol = pred.sum(dim=[2,3,4]) * voxel_vol
    expected_vol = info['growth_rate'].to(pred_vol.device) * t
    growth_penalty = torch.mean((pred_vol - expected_vol).pow(2))
    return change_penalty + 0.01 * growth_penalty

# ---------------------- Utils ----------------------
def validate(dataloader, encoder, decoder, vq, odeblock, discriminator, device, temporal=False, gan=False):
    encoder.eval(); decoder.eval(); vq.eval(); odeblock.eval(); discriminator.eval()
    metrics = {'recon':0, 'vq':0, 'phys':0, 'd_real':0, 'd_fake':0}; count=0
    with torch.no_grad():
        for vol, info, tp in dataloader:
            vol, tp = vol.to(device), tp.to(device)
            z = encoder(vol); q, vl = vq(z)
            if temporal:
                tpts = torch.stack([torch.zeros_like(tp), tp],0)
                q = odeblock(q, tpts)
                pl = compute_physics_loss(decoder(q), info, tp)
                metrics['phys'] += pl.item()
            recon = decoder(q)
            rl = nn.functional.mse_loss(recon, vol)
            metrics['recon'] += rl.item(); metrics['vq'] += vl.item()
            if gan:
                metrics['d_real'] += torch.mean(discriminator(vol)).item()
                metrics['d_fake'] += torch.mean(discriminator(recon)).item()
            count+=1
    for k in metrics: metrics[k]/=count
    encoder.train(); decoder.train(); vq.train(); odeblock.train(); discriminator.train()
    return metrics

def plot_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for key, vals in history.items():
        plt.figure(); plt.plot(vals); plt.title(key)
        plt.savefig(os.path.join(save_dir, f'{key}.png'))
        plt.close()

# ---------------------- Training Loops ----------------------
def train_basic(args):
    # Prepare
    train_ds = LongitudinalMRIDataset(args.train_csv)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_dl = DataLoader(LongitudinalMRIDataset(args.val_csv), batch_size=args.bs) if args.val_csv else None

    device = torch.device(args.device)
    enc, dec = Encoder3D().to(device), Decoder3D().to(device)
    vq = VectorQuantizer(512,64).to(device)
    disc = Discriminator3D().to(device)
    opt_g = optim.Adam(list(enc.parameters())+list(dec.parameters())+list(vq.parameters()), lr=args.lr)
    opt_d = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5,0.9))

    writer = SummaryWriter(args.logdir)
    history = {'train_recon':[], 'train_vq':[], 'train_d_real':[], 'train_d_fake':[],
               'val_recon':[], 'val_vq':[], 'val_d_real':[], 'val_d_fake':[]}

    for ep in range(args.epochs):
        sum_r = sum_v = sum_dr = sum_df = 0
        for vol, _, _ in train_dl:
            vol = vol.to(device)
            # Discriminator step
            with torch.no_grad():
                z = enc(vol); q, _ = vq(z)
                fake = dec(q)
            pred_real = disc(vol); pred_fake = disc(fake)
            loss_d = torch.mean(nn.functional.relu(1.0 - pred_real)) + torch.mean(nn.functional.relu(1.0 + pred_fake))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # Generator step
            z = enc(vol); q, vl = vq(z); recon = dec(q)
            rl = nn.functional.mse_loss(recon, vol)
            pred_fake_for_g = disc(recon)
            loss_gan = -torch.mean(pred_fake_for_g)
            loss_g = rl + vl + args.lambda_gan * loss_gan
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            sum_r += rl.item(); sum_v += vl.item()
            sum_dr += torch.mean(pred_real).item(); sum_df += torch.mean(pred_fake).item()

        # Logging
        avg_r, avg_v = sum_r/len(train_dl), sum_v/len(train_dl)
        avg_dr, avg_df = sum_dr/len(train_dl), sum_df/len(train_dl)
        history['train_recon'].append(avg_r); history['train_vq'].append(avg_v)
        history['train_d_real'].append(avg_dr); history['train_d_fake'].append(avg_df)
        writer.add_scalar('Train/Reconstruction', avg_r, ep)
        writer.add_scalar('Train/VQ', avg_v, ep)
        writer.add_scalar('Train/D_real', avg_dr, ep)
        writer.add_scalar('Train/D_fake', avg_df, ep)

        if val_dl:
            m = validate(val_dl, enc, dec, vq, ODEBlock(ODEFunc(1)), disc, device, temporal=False, gan=True)
            history['val_recon'].append(m['recon']); history['val_vq'].append(m['vq'])
            history['val_d_real'].append(m['d_real']); history['val_d_fake'].append(m['d_fake'])
            writer.add_scalar('Val/Reconstruction', m['recon'], ep)
            writer.add_scalar('Val/VQ', m['vq'], ep)
            writer.add_scalar('Val/D_real', m['d_real'], ep)
            writer.add_scalar('Val/D_fake', m['d_fake'], ep)

        print(f"[Basic GAN][Epoch {ep}] Recon={avg_r:.4f} VQ={avg_v:.4f} D_real={avg_dr:.4f} D_fake={avg_df:.4f}")

    writer.close(); plot_history(history, args.save_dir)


def train_temporal(args):
    train_ds = LongitudinalMRIDataset(args.train_csv)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_dl = DataLoader(LongitudinalMRIDataset(args.val_csv), batch_size=args.bs) if args.val_csv else None

    device = torch.device(args.device)
    enc, dec = Encoder3D().to(device), Decoder3D().to(device)
    vq = VectorQuantizer(512,64).to(device)
    disc = Discriminator3D().to(device)

    # latent dim
    dummy = torch.zeros(1,64,32,32,64)
    latent_dim = dummy.numel()
    func = ODEFunc(latent_dim).to(device)
    odeb = ODEBlock(func).to(device)

    opt_g = optim.Adam(list(enc.parameters())+list(dec.parameters())+list(vq.parameters())+list(func.parameters()), lr=args.lr)
    opt_d = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5,0.9))
    writer = SummaryWriter(args.logdir)

    history = {'train_recon':[], 'train_vq':[], 'train_phys':[], 'train_d_real':[], 'train_d_fake':[],
               'val_recon':[], 'val_vq':[], 'val_phys':[], 'val_d_real':[], 'val_d_fake':[]}

    for ep in range(args.epochs):
        sum_r=sum_v=sum_p=sum_dr=sum_df=0
        for vol, info, tp in train_dl:
            vol, tp = vol.to(device), tp.to(device)
            # 1) Discriminator on real vs fake
            with torch.no_grad():
                z = enc(vol); q, _ = vq(z)
                tpts = torch.stack([torch.zeros_like(tp), tp],0)
                lat = odeb(q, tpts)
                fake = dec(lat)
            pred_real = disc(vol); pred_fake = disc(fake)
            loss_d = torch.mean(nn.functional.relu(1.0 - pred_real)) + torch.mean(nn.functional.relu(1.0 + pred_fake))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # 2) Generator step
            z = enc(vol); q, vl = vq(z)
            tpts = torch.stack([torch.zeros_like(tp), tp],0)
            lat = odeb(q, tpts)
            recon = dec(lat)

            rl = nn.functional.mse_loss(recon, vol)
            pl = compute_physics_loss(recon, info, tp)
            pred_fake_for_g = disc(recon)
            loss_gan = -torch.mean(pred_fake_for_g)
            loss_g = rl + vl + 0.1*pl + args.lambda_gan * loss_gan

            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            sum_r+=rl.item(); sum_v+=vl.item(); sum_p+=pl.item()
            sum_dr+=torch.mean(pred_real).item(); sum_df+=torch.mean(pred_fake).item()

        # Logging
        avg_r,avg_v,avg_p = sum_r/len(train_dl), sum_v/len(train_dl), sum_p/len(train_dl)
        avg_dr,avg_df = sum_dr/len(train_dl), sum_df/len(train_dl)
        history['train_recon'].append(avg_r); history['train_vq'].append(avg_v)
        history['train_phys'].append(avg_p); history['train_d_real'].append(avg_dr)
        history['train_d_fake'].append(avg_df)
        writer.add_scalar('Train/Reconstruction', avg_r, ep)
        writer.add_scalar('Train/VQ', avg_v, ep)
        writer.add_scalar('Train/Physics', avg_p, ep)
        writer.add_scalar('Train/D_real', avg_dr, ep)
        writer.add_scalar('Train/D_fake', avg_df, ep)

        if val_dl:
            m = validate(val_dl, enc, dec, vq, odeb, disc, device, temporal=True, gan=True)
            history['val_recon'].append(m['recon']); history['val_vq'].append(m['vq'])
            history['val_phys'].append(m['phys']); history['val_d_real'].append(m['d_real'])
            history['val_d_fake'].append(m['d_fake'])
            writer.add_scalar('Val/Reconstruction', m['recon'], ep)
            writer.add_scalar('Val/VQ', m['vq'], ep)
            writer.add_scalar('Val/Physics', m['phys'], ep)
            writer.add_scalar('Val/D_real', m['d_real'], ep)
            writer.add_scalar('Val/D_fake', m['d_fake'], ep)

        print(f"[Temporal GAN][Epoch {ep}] Recon={avg_r:.4f} VQ={avg_v:.4f} Phys={avg_p:.4f} D_real={avg_dr:.4f} D_fake={avg_df:.4f}")

    writer.close(); plot_history(history, args.save_dir)

# ---------------------- Entry Point ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','train_temporal'], default='train_temporal')
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', default=None)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='runs/exp')
    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for adversarial loss')
    args = parser.parse_args()
    if args.mode == 'train':
        train_basic(args)
    else:
        train_temporal(args)
