#!/usr/bin/env python
"""
4D VQ-GAN for Reconstruction on BraTS 2023 and Temporal Modeling on LUMIERE
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

# ----------------------------------------
# Module‐level default for physics loss weight
# ----------------------------------------
GROWTH_PENALTY_WEIGHT = 0.01

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
        vol = torch.from_numpy(img).unsqueeze(0)  # [1, H, W, D]

        # pad depth so D % 4 == 0
        C, H, W, D = vol.shape
        pad_d = (4 - (D % 4)) % 4
        if pad_d:
            zeros = torch.zeros(C, H, W, pad_d, dtype=vol.dtype)
            vol = torch.cat([vol, zeros], dim=3)

        # normalize intensities to [0,1]
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)

        if self.transform:
            vol = self.transform(vol)

        return vol, {'patient_id': pid, 'growth_rate': torch.tensor(0.05)}, torch.tensor(timepoint)


class LumiereDataset(Dataset):
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
        vol = torch.from_numpy(img).unsqueeze(0)

        # normalize intensities to [0,1]
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)

        if self.transform:
            vol = self.transform(vol)

        return vol, {'patient_id': pid, 'growth_rate': torch.tensor(0.05)}, torch.tensor(timepoint)

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


class Discriminator3D(nn.Module):
    def __init__(self, in_ch=1, features=[64,128,256]):
        super().__init__()
        layers = []
        prev = in_ch
        for f in features:
            layers += [nn.Conv3d(prev, f, 4, 2, 1),
                       nn.LeakyReLU(0.2, inplace=True)]
            prev = f
        layers += [nn.Conv3d(prev, 1, 4, 1, 0)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
                                 method=self.method,
                                 atol=self.tol, rtol=self.tol)
        return out[-1].reshape(B, C, H, W, D)


# ---------------------- Physics-Informed Loss ----------------------
def compute_physics_loss(pred, info, t):
    diff = pred - pred.detach()
    change_penalty = torch.mean(torch.abs(diff) / (t.view(-1,1,1,1,1) + 1e-6))
    voxel_vol = 1.0
    pred_vol = pred.sum(dim=[2,3,4]) * voxel_vol
    expected_vol = info['growth_rate'].to(pred_vol.device) * t
    growth_penalty = torch.mean((pred_vol - expected_vol).pow(2))
    return change_penalty + GROWTH_PENALTY_WEIGHT * growth_penalty


# ---------------------- Utility ----------------------
def plot_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for key, vals in history.items():
        plt.figure(); plt.plot(vals); plt.title(key)
        plt.savefig(os.path.join(save_dir, f'{key}.png'))
        plt.close()


# ---------------------- Training & Validation ----------------------
def train_recon(args):
    dl      = DataLoader(Brats2023Dataset(args.train_csv),
                         batch_size=args.bs, shuffle=True)
    val_dl  = (DataLoader(Brats2023Dataset(args.val_csv),
                         batch_size=args.bs)
               if args.val_csv else None)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    enc, dec = Encoder3D().to(device), Decoder3D().to(device)
    vq       = VectorQuantizer(512,64, commitment_cost=args.commitment_cost).to(device)
    disc     = Discriminator3D().to(device)

    opt_g = optim.Adam(
        list(enc.parameters())+
        list(dec.parameters())+
        list(vq.parameters()),
        lr=args.lr
    )
    opt_d = optim.Adam(
        disc.parameters(),
        lr=args.disc_lr,
        betas=(0.5,0.9)
    )

    writer  = SummaryWriter(args.logdir)
    history = {
        'train_recon':[], 'train_vq':[],
        'train_d_real':[], 'train_d_fake':[],
        'val_recon':[],   'val_vq':[],
        'val_d_real':[],  'val_d_fake':[]
    }

    for ep in range(args.epochs):
        is_pre = (ep < args.pretrain_epochs)
        phase  = "Pretrain" if is_pre else "Adversarial"
        print(f"\n=== {phase} Recon Epoch {ep+1}/{args.epochs} ===")

        sum_r = sum_v = sum_dr = sum_df = 0
        for bi, (vol, _, _) in enumerate(dl, 1):
            vol = vol.to(device)

            # Discriminator step
            if (not is_pre) and (bi % args.disc_freq == 0):
                with torch.no_grad():
                    z = enc(vol); q, _ = vq(z)
                    fake = dec(q)
                p_real, p_fake = disc(vol), disc(fake)
                loss_d = (torch.mean(nn.functional.relu(1-p_real))
                          + torch.mean(nn.functional.relu(1+p_fake)))
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()
                sum_dr += p_real.mean().item()
                sum_df += p_fake.mean().item()

            # Generator step
            z    = enc(vol)
            q,vl = vq(z)
            recon= dec(q)
            rl   = nn.functional.mse_loss(recon, vol)
            gan_term = 0.0
            if not is_pre:
                gan_term = -disc(recon).mean() * args.lambda_gan
            loss_g = rl + vl + gan_term
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            sum_r += rl.item()
            sum_v += vl.item()

            if args.batch_verbose and (bi % args.batch_verbose == 0):
                msg = f"  [B{bi}/{len(dl)}] R={sum_r/bi:.4f} VQ={sum_v/bi:.4f}"
                if not is_pre:
                    c = bi//args.disc_freq
                    msg += f" D_r={sum_dr/max(1,c):.4f} D_f={sum_df/max(1,c):.4f}"
                print(msg)

        dr_norm = sum_dr/(len(dl)//args.disc_freq or 1)
        df_norm = sum_df/(len(dl)//args.disc_freq or 1)
        print(f"=> Recon Ep{ep+1}: R={sum_r/len(dl):.4f} VQ={sum_v/len(dl):.4f}"
              + (f" D_r={dr_norm:.4f} D_f={df_norm:.4f}" if not is_pre else ""))

        # validation
        if val_dl:
            m = validate(val_dl, enc, dec, vq, ODEBlock(ODEFunc(1)), disc,
                         device, temporal=False, gan=True)
            for k in ['recon','vq','d_real','d_fake']:
                history[f'val_{k}'].append(m[k])
                writer.add_scalar(f'Val/{k.capitalize()}', m[k], ep)

        history['train_recon'].append(sum_r/len(dl))
        history['train_vq'].append(sum_v/len(dl))
        history['train_d_real'].append(dr_norm)
        history['train_d_fake'].append(df_norm)
        writer.add_scalar('Train/Reconstruction', history['train_recon'][-1], ep)
        writer.add_scalar('Train/VQ',             history['train_vq'][-1],    ep)
        writer.add_scalar('Train/D_real',         history['train_d_real'][-1], ep)
        writer.add_scalar('Train/D_fake',         history['train_d_fake'][-1], ep)

        # checkpoint
        if ((ep+1) % args.save_ckpt_freq) == 0:
            ckpt = {
                'epoch': ep+1,
                'enc':   enc.state_dict(),
                'dec':   dec.state_dict(),
                'vq':    vq.state_dict(),
                'disc':  disc.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict()
            }
            torch.save(ckpt,
                       os.path.join(args.save_dir,
                                    f'checkpoint_recon_ep{ep+1}.pth'))

    writer.close()
    plot_history(history, args.save_dir)


def train_temporal(args):
    dl     = DataLoader(LumiereDataset(args.train_csv),
                       batch_size=args.bs, shuffle=True)
    val_dl = (DataLoader(LumiereDataset(args.val_csv), batch_size=args.bs)
              if args.val_csv else None)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    enc, dec = Encoder3D().to(device), Decoder3D().to(device)
    vq       = VectorQuantizer(512,64, commitment_cost=args.commitment_cost).to(device)
    disc     = Discriminator3D().to(device)

    dummy = torch.zeros(1,64,32,32,64)
    latent_dim = dummy.numel()
    func = ODEFunc(latent_dim).to(device)
    odeb = ODEBlock(func).to(device)

    opt_g = optim.Adam(
        list(enc.parameters())+
        list(dec.parameters())+
        list(vq.parameters())+
        list(func.parameters()),
        lr=args.lr
    )
    opt_d = optim.Adam(
        disc.parameters(),
        lr=args.disc_lr,
        betas=(0.5,0.9)
    )

    writer  = SummaryWriter(args.logdir)
    history = {
        'train_recon':[], 'train_vq':[],  'train_phys':[],
        'train_d_real':[], 'train_d_fake':[],
        'val_recon':[],   'val_vq':[],   'val_phys':[],
        'val_d_real':[],  'val_d_fake':[]
    }

    for ep in range(args.epochs):
        is_pre = (ep < args.pretrain_epochs)
        phase  = "Pretrain" if is_pre else "Adversarial"
        print(f"\n=== {phase} Temporal Epoch {ep+1}/{args.epochs} ===")

        sum_r=sum_v=sum_p=sum_dr=sum_df=0
        for bi, (vol, info, tp) in enumerate(dl, 1):
            vol, tp = vol.to(device), tp.to(device)

            # Discriminator
            if (not is_pre) and (bi % args.disc_freq == 0):
                with torch.no_grad():
                    z = enc(vol); q, _ = vq(z)
                    tpts = torch.stack([torch.zeros_like(tp), tp], 0)
                    lat = odeb(q, tpts)
                    fake = dec(lat)
                p_real, p_fake = disc(vol), disc(fake)
                loss_d = (torch.mean(nn.functional.relu(1-p_real))
                          + torch.mean(nn.functional.relu(1+p_fake)))
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()
                sum_dr += p_real.mean().item()
                sum_df += p_fake.mean().item()

            # Generator
            z    = enc(vol)
            q,vl = vq(z)
            tpts = torch.stack([torch.zeros_like(tp), tp], 0)
            lat  = odeb(q, tpts)
            recon= dec(lat)

            rl = nn.functional.mse_loss(recon, vol)
            pl = compute_physics_loss(recon, info, tp)
            gan_term = 0.0
            if not is_pre:
                gan_term = -disc(recon).mean() * args.lambda_gan

            loss_g = rl + vl + args.physics_weight * pl + gan_term
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            sum_r += rl.item(); sum_v += vl.item(); sum_p += pl.item()

            if args.batch_verbose and (bi % args.batch_verbose == 0):
                msg = (f"  [B{bi}/{len(dl)}] R={sum_r/bi:.4f} "
                       f"VQ={sum_v/bi:.4f} P={sum_p/bi:.4f}")
                if not is_pre:
                    c = bi//args.disc_freq
                    msg += f" D_r={sum_dr/max(1,c):.4f} D_f={sum_df/max(1,c):.4f}"
                print(msg)

        dr_norm = sum_dr/(len(dl)//args.disc_freq or 1)
        df_norm = sum_df/(len(dl)//args.disc_freq or 1)
        print(f"=> Temporal Ep{ep+1}: R={sum_r/len(dl):.4f} "
              f"VQ={sum_v/len(dl):.4f} P={sum_p/len(dl):.4f}"
              + (f" D_r={dr_norm:.4f} D_f={df_norm:.4f}" if not is_pre else ""))

        # validation
        if val_dl:
            m = validate(val_dl, enc, dec, vq, odeb, disc,
                         device, temporal=True, gan=True)
            for k in ['recon','vq','phys','d_real','d_fake']:
                history[f'val_{k}'].append(m[k])
                writer.add_scalar(f'Val/{k.capitalize()}', m[k], ep)

        history['train_recon'].append(sum_r/len(dl))
        history['train_vq'].append(sum_v/len(dl))
        history['train_phys'].append(sum_p/len(dl))
        history['train_d_real'].append(dr_norm)
        history['train_d_fake'].append(df_norm)
        writer.add_scalar('Train/Reconstruction', history['train_recon'][-1], ep)
        writer.add_scalar('Train/VQ',             history['train_vq'][-1],    ep)
        writer.add_scalar('Train/Physics',        history['train_phys'][-1],   ep)
        writer.add_scalar('Train/D_real',         history['train_d_real'][-1], ep)
        writer.add_scalar('Train/D_fake',         history['train_d_fake'][-1], ep)

        # checkpoint
        if ((ep+1) % args.save_ckpt_freq) == 0:
            ckpt = {
                'epoch': ep+1,
                'enc':      enc.state_dict(),
                'dec':      dec.state_dict(),
                'vq':       vq.state_dict(),
                'disc':     disc.state_dict(),
                'odefunc':  func.state_dict(),
                'opt_g':    opt_g.state_dict(),
                'opt_d':    opt_d.state_dict()
            }
            torch.save(ckpt,
                       os.path.join(args.save_dir,
                                    f'checkpoint_temp_ep{ep+1}.pth'))

    writer.close()
    plot_history(history, args.save_dir)


def validate(dataloader, encoder, decoder, vq, odeblock, discriminator,
             device, temporal=False, gan=False):
    encoder.eval(); decoder.eval(); vq.eval()
    odeblock.eval(); discriminator.eval()
    metrics = dict.fromkeys(['recon','vq','phys','d_real','d_fake'], 0.0)
    cnt = 0
    with torch.no_grad():
        for vol, info, tp in dataloader:
            vol, tp = vol.to(device), tp.to(device)
            z = encoder(vol); q, vl = vq(z)
            if temporal:
                tpts = torch.stack([torch.zeros_like(tp), tp], 0)
                q = odeblock(q, tpts)
                pl = compute_physics_loss(decoder(q), info, tp)
                metrics['phys'] += pl.item()
            recon = decoder(q)
            metrics['recon'] += nn.functional.mse_loss(recon, vol).item()
            metrics['vq']    += vl.item()
            if gan:
                metrics['d_real'] += discriminator(vol).mean().item()
                metrics['d_fake'] += discriminator(recon).mean().item()
            cnt += 1
    for k in metrics: metrics[k] /= cnt
    encoder.train(); decoder.train(); vq.train()
    odeblock.train(); discriminator.train()
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train_recon','train_temporal'], default='train_recon')
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', default=None)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--disc_lr', type=float, default=1e-3)
    parser.add_argument('--disc_freq', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--pretrain_epochs', type=int, default=0)
    parser.add_argument('--save_ckpt_freq', type=int, default=1)
    parser.add_argument('--lambda_gan', type=float, default=1.0)
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--growth_penalty_weight', type=float, default=0.01,
                        help="weight for physics-based volume loss")
    parser.add_argument('--physics_weight', type=float, default=0.1,
                        help="multiplier for physics loss in temporal mode")
    parser.add_argument('--batch_verbose', type=int, default=0,
                        help="print every N batches if >0")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='runs/exp')
    parser.add_argument('--save_dir', type=str, default='outputs')
    args = parser.parse_args()

    # override module-level physics weight
    GROWTH_PENALTY_WEIGHT = args.growth_penalty_weight

    if args.mode == 'train_recon':
        train_recon(args)
    else:
        train_temporal(args)
