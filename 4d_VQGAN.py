"""
4D VQ-GAN for Reconstruction on BraTS 2023 and Temporal Modeling on LUMIERE

This script provides:
 1. BraTS2023Dataset loader for static reconstruction training
 2. LumiereDataset loader for longitudinal temporal training
 3. 3D VQ-GAN components (Encoder, Decoder, VectorQuantizer)
 4. Neural ODE temporal modeling (ODEFunc, ODEBlock)
 5. Physics-informed loss
 6. 3D Discriminator (PatchGAN style)
 7. Training loops: reconstruction & temporal + physics + GAN
 8. TensorBoard logging, validation, and loss-curve plotting

Usage:
  # Reconstruction on BraTS 2023
  python fourd_vqgan_modified.py --mode train_recon \
                                 --train_csv brats_train.csv \
                                 --val_csv brats_val.csv \
                                 --logdir runs/recon \
                                 --save_dir outputs/recon

  # Temporal on LUMIERE
  python fourd_vqgan_modified.py --mode train_temporal \
                                 --train_csv lumiere_train.csv \
                                 --val_csv lumiere_val.csv \
                                 --logdir runs/temporal \
                                 --save_dir outputs/temporal
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

# ---------------------- Dataset Loaders ----------------------
class Brats2023Dataset(Dataset):
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
        vol = torch.from_numpy(img).unsqueeze(0)  # [C=1, H, W, D]

        # --- pad depth so D % 4 == 0 ---
        _, H, W, D = vol.shape
        pad_d = (4 - (D % 4)) % 4   # if D%4 !=0, pad_d = 4 - (D%4)
        if pad_d:
            # create zeros of shape [1,H,W,pad_d]
            zeros = torch.zeros(vol.shape[0], H, W, pad_d,
                                dtype=vol.dtype, device=vol.device)
            # concatenate along depth axis (last dim)
            vol = torch.cat([vol, zeros], dim=3)
        # ----------------------------------

        if self.transform:
            vol = self.transform(vol)

        # your growth_rate etc can be set here or passed through info dict
        info = {'patient_id': pid, 'growth_rate': 0.05}
        return vol, info, torch.tensor(timepoint)

class LumiereDataset(Dataset):
    """Longitudinal dataset loader for LUMIERE temporal modeling"""
    def __init__(self, csv_path, transform=None):
        self.entries = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for pid, tp, img_path in reader:
                self.entries.append((pid, float(tp), img_path))
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, timepoint, img_path = self.entries[idx]
        img = nib.load(img_path).get_fdata().astype('float32')
        vol = torch.from_numpy(img).unsqueeze(0)  # [1,H,W,D]
        if self.transform:
            vol = self.transform(vol)
        # info includes dummy growth_rate (could be replaced with real values)
        return vol, {'patient_id': pid, 'growth_rate': 0.05}, torch.tensor(timepoint)


# ---------------------- VQ-GAN Components ----------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        B, C, H, W, D = x.shape
        flat = x.permute(0,2,3,4,1).reshape(-1, C)
        dist = (flat.pow(2).sum(1,keepdim=True)
               - 2*flat @ self.embedding.weight.t()
               + self.embedding.weight.pow(2).sum(1))
        idx = torch.argmin(dist, dim=1)
        enc = torch.zeros(idx.size(0), self.embedding.num_embeddings, device=x.device)
        enc.scatter_(1, idx.unsqueeze(1), 1)
        quant_flat = enc @ self.embedding.weight
        quant = quant_flat.view(B, H, W, D, C).permute(0,4,1,2,3)
        e_loss = nn.functional.mse_loss(quant.detach(), x)
        q_loss = nn.functional.mse_loss(quant, x.detach())
        loss = q_loss + self.commitment_cost * e_loss
        quant = x + (quant - x).detach()
        return quant, loss


class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, hidden=128, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(hidden, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(hidden, emb, 3, 1, 1)
        )
    def forward(self, x): return self.net(x)


class Decoder3D(nn.Module):
    def __init__(self, emb=64, hidden=128, out=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(emb, hidden, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose3d(hidden, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(hidden, out, 3, 1, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)


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
            nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim)
        )
    def forward(self, t, z): return self.net(z)


class ODEBlock(nn.Module):
    def __init__(self, func, method='rk4', tol=1e-3):
        super().__init__()
        self.func, self.method, self.tol = func, method, tol

    def forward(self, z, timepts):
        B, C, H, W, D = z.shape
        flat = z.reshape(B, -1)
        out = torchdiffeq.odeint(self.func, flat, timepts.to(flat.device),
                                 method=self.method, atol=self.tol, rtol=self.tol)
        return out[-1].reshape(B, C, H, W, D)


# ---------------------- Physics-Informed Loss ----------------------
def compute_physics_loss(pred, info, t):
    diff = pred - pred.detach()
    change_penalty = torch.mean(torch.abs(diff) / (t.view(-1,1,1,1,1) + 1e-6))
    voxel_vol = 1.0
    pred_vol = pred.sum(dim=[2,3,4]) * voxel_vol
    expected_vol = info['growth_rate'].to(pred_vol.device) * t
    growth_penalty = torch.mean((pred_vol - expected_vol).pow(2))
    return change_penalty + 0.01 * growth_penalty


# ---------------------- Validation & Utilities ----------------------
def validate(dataloader, encoder, decoder, vq, odeblock, discriminator,
             device, temporal=False, gan=False):
    encoder.eval(); decoder.eval(); vq.eval(); odeblock.eval(); discriminator.eval()
    metrics = {k:0 for k in ['recon','vq','phys','d_real','d_fake']}; count=0
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
            metrics['recon'] += nn.functional.mse_loss(recon, vol).item()
            metrics['vq'] += vl.item()
            if gan:
                metrics['d_real'] += discriminator(vol).mean().item()
                metrics['d_fake'] += discriminator(recon).mean().item()
            count += 1
    for k in metrics: metrics[k] /= count
    encoder.train(); decoder.train(); vq.train(); odeblock.train(); discriminator.train()
    return metrics


def plot_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for key, vals in history.items():
        plt.figure(); plt.plot(vals); plt.title(key)
        plt.savefig(os.path.join(save_dir, f'{key}.png'))
        plt.close()


# ---------------------- Training Loops ----------------------
def train_recon(args):
    train_ds = Brats2023Dataset(args.train_csv)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_dl = DataLoader(Brats2023Dataset(args.val_csv), batch_size=args.bs) if args.val_csv else None

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
        sum_r=sum_v=sum_dr=sum_df=0
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
            loss_gan = -disc(recon).mean()
            loss_g = rl + vl + args.lambda_gan * loss_gan
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            sum_r += rl.item(); sum_v += vl.item()
            sum_dr += pred_real.mean().item(); sum_df += pred_fake.mean().item()

        # Logging
        history['train_recon'].append(sum_r/len(train_dl))
        history['train_vq'].append(sum_v/len(train_dl))
        history['train_d_real'].append(sum_dr/len(train_dl))
        history['train_d_fake'].append(sum_df/len(train_dl))
        writer.add_scalar('Train/Reconstruction', history['train_recon'][-1], ep)
        writer.add_scalar('Train/VQ', history['train_vq'][-1], ep)
        writer.add_scalar('Train/D_real', history['train_d_real'][-1], ep)
        writer.add_scalar('Train/D_fake', history['train_d_fake'][-1], ep)

        if val_dl:
            m = validate(val_dl, enc, dec, vq, ODEBlock(ODEFunc(1)), disc, device, temporal=False, gan=True)
            for k in ['recon','vq','d_real','d_fake']:
                history[f'val_{k}'].append(m[k])
                writer.add_scalar(f'Val/{k.capitalize()}', m[k], ep)

        print(f"[Recon][Epoch {ep}] R={history['train_recon'][-1]:.4f} VQ={history['train_vq'][-1]:.4f} D_r={history['train_d_real'][-1]:.4f} D_f={history['train_d_fake'][-1]:.4f}")

    writer.close()
    plot_history(history, args.save_dir)


def train_temporal(args):
    train_ds = LumiereDataset(args.train_csv)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_dl = DataLoader(LumiereDataset(args.val_csv), batch_size=args.bs) if args.val_csv else None

    device = torch.device(args.device)
    enc, dec = Encoder3D().to(device), Decoder3D().to(device)
    vq = VectorQuantizer(512,64).to(device)
    disc = Discriminator3D().to(device)

    # latent dimension calculation
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
            # Discriminator
            with torch.no_grad():
                z = enc(vol); q, _ = vq(z)
                tpts = torch.stack([torch.zeros_like(tp), tp], 0)
                lat = odeb(q, tpts)
                fake = dec(lat)
            pred_real = disc(vol); pred_fake = disc(fake)
            loss_d = torch.mean(nn.functional.relu(1.0 - pred_real)) + torch.mean(nn.functional.relu(1.0 + pred_fake))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # Generator
            z = enc(vol); q, vl = vq(z)
            tpts = torch.stack([torch.zeros_like(tp), tp], 0)
            lat = odeb(q, tpts)
            recon = dec(lat)

            rl = nn.functional.mse_loss(recon, vol)
            pl = compute_physics_loss(recon, info, tp)
            loss_gan = -disc(recon).mean()
            loss_g = rl + vl + 0.1 * pl + args.lambda_gan * loss_gan

            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            sum_r += rl.item(); sum_v += vl.item(); sum_p += pl.item()
            sum_dr += pred_real.mean().item(); sum_df += pred_fake.mean().item()

        history['train_recon'].append(sum_r/len(train_dl))
        history['train_vq'].append(sum_v/len(train_dl))
        history['train_phys'].append(sum_p/len(train_dl))
        history['train_d_real'].append(sum_dr/len(train_dl))
        history['train_d_fake'].append(sum_df/len(train_dl))

        writer.add_scalar('Train/Reconstruction', history['train_recon'][-1], ep)
        writer.add_scalar('Train/VQ', history['train_vq'][-1], ep)
        writer.add_scalar('Train/Physics', history['train_phys'][-1], ep)
        writer.add_scalar('Train/D_real', history['train_d_real'][-1], ep)
        writer.add_scalar('Train/D_fake', history['train_d_fake'][-1], ep)

        if val_dl:
            m = validate(val_dl, enc, dec, vq, odeb, disc, device, temporal=True, gan=True)
            for k in ['recon','vq','phys','d_real','d_fake']:
                history[f'val_{k}'].append(m[k])
                writer.add_scalar(f'Val/{k.capitalize()}', m[k], ep)

        print(f"[Temporal][Epoch {ep}] R={history['train_recon'][-1]:.4f} VQ={history['train_vq'][-1]:.4f} P={history['train_phys'][-1]:.4f} D_r={history['train_d_real'][-1]:.4f} D_f={history['train_d_fake'][-1]:.4f}")

    writer.close()
    plot_history(history, args.save_dir)


# ---------------------- Entry Point ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train_recon','train_temporal'], default='train_recon')
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
    if args.mode == 'train_recon':
        train_recon(args)
    else:
        train_temporal(args)
