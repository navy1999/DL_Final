#!/usr/bin/env python
"""
4D VQ-GAN for Reconstruction on BraTS 2023 and Temporal Modeling on LUMIERE
(with reduced memory footprint via smaller channels, checkpointed decoder, AMP, and auto-cropping)
"""
import os, argparse, csv
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import torchdiffeq
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# module‐level default (overridden by --growth_penalty_weight)
GROWTH_PENALTY_WEIGHT = 0.01

# ---------------------- Dataset Loaders ----------------------
class Brats2023Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.entries = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader, None)
            for pid, tp, path in reader:
                self.entries.append((pid, float(tp), path))
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, tp, path = self.entries[idx]
        img = nib.load(path).get_fdata().astype('float32')
        vol = torch.from_numpy(img).unsqueeze(0)  # [C=1, H, W, D]

        # --- NEW PADDING LOGIC ---
        # Pad H, W, D up to a multiple of 4 AND ensure each >= 4
        C, H, W, D = vol.shape
        pad_h = (4 - (H % 4)) % 4
        if H + pad_h < 4:
            pad_h = 4 - H
        pad_w = (4 - (W % 4)) % 4
        if W + pad_w < 4:
            pad_w = 4 - W
        pad_d = (4 - (D % 4)) % 4
        if D + pad_d < 4:
            pad_d = 4 - D

        if pad_h or pad_w or pad_d:
            # For a 4D tensor [C,H,W,D], pad order is
            # (D_left, D_right, W_left, W_right, H_left, H_right)
            vol = nn.functional.pad(vol,
                                    (0, pad_d,    # pad D
                                     0, pad_w,    # pad W
                                     0, pad_h),   # pad H
                                    mode='constant', value=0.0)

        # normalize
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
        if self.transform:
            vol = self.transform(vol)

        return vol, {'patient_id': pid, 'growth_rate': torch.tensor(0.05)}, torch.tensor(tp)


class LumiereDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.entries = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader, None)
            for pid, tp, path in reader:
                self.entries.append((pid, float(tp), path))
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, tp, path = self.entries[idx]
        img = nib.load(path).get_fdata().astype('float32')
        vol = torch.from_numpy(img).unsqueeze(0)

        # --- NEW PADDING LOGIC ---
        C, H, W, D = vol.shape
        pad_h = (4 - (H % 4)) % 4
        if H + pad_h < 4:
            pad_h = 4 - H
        pad_w = (4 - (W % 4)) % 4
        if W + pad_w < 4:
            pad_w = 4 - W
        pad_d = (4 - (D % 4)) % 4
        if D + pad_d < 4:
            pad_d = 4 - D

        if pad_h or pad_w or pad_d:
            vol = nn.functional.pad(vol,
                                    (0, pad_d,
                                     0, pad_w,
                                     0, pad_h),
                                    mode='constant', value=0.0)

        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
        if self.transform:
            vol = self.transform(vol)

        return vol, {'patient_id': pid, 'growth_rate': torch.tensor(0.05)}, torch.tensor(tp)


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
        dist = flat.pow(2).sum(1, keepdim=True) \
               - 2*flat @ self.embedding.weight.t() \
               + self.embedding.weight.pow(2).sum(1)
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
    def __init__(self, in_ch=1, hidden=64, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch,   hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(hidden,  hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(hidden,  emb,    3, 1, 1),
        )
    def forward(self, x): return self.net(x)


class Decoder3D(nn.Module):
    def __init__(self, emb=64, hidden=64, out=1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose3d(emb, hidden, 4, 2, 1),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose3d(hidden, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv3d(hidden, out, 3, 1, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = checkpoint(self.block1, x)
        return self.block2(x)


class Discriminator3D(nn.Module):
    def __init__(self, in_ch=1, features=[64,128,256]):
        super().__init__()
        layers, prev = [], in_ch
        for f in features:
            layers += [nn.Conv3d(prev, f, 4, 2, 1),
                       nn.LeakyReLU(0.2, inplace=True)]
            prev = f
        layers += [nn.Conv3d(prev, 1, 4, 1, 0)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


# ---------------------- ODE Components ----------------------
class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*2), nn.ReLU(),
            nn.Linear(dim*2, dim)
        )
    def forward(self, t, z): return self.net(z)


class ODEBlock(nn.Module):
    def __init__(self, func, method='rk4', tol=1e-3):
        super().__init__()
        self.func, self.method, self.tol = func, method, tol
    def forward(self, z, timepts):
        if timepts[0] == timepts[-1]:
            return z
        B,C,H,W,D = z.shape
        flat = z.permute(0,2,3,4,1).reshape(-1, C)
        t = timepts.to(flat.device)
        if t[1] <= t[0]:
            t = t + torch.tensor([0.0,1e-6], device=t.device)
        out = torchdiffeq.odeint(self.func, flat, t,
                                 method=self.method,
                                 atol=self.tol, rtol=self.tol)
        return out[-1].reshape(B,H,W,D,C).permute(0,4,1,2,3)


# ---------------------- Loss & Utility ----------------------
def compute_physics_loss(pred, info, t):
    diff = pred - pred.detach()
    change = torch.mean(torch.abs(diff) / (t.view(-1,1,1,1,1)+1e-6))
    vol_pred = pred.sum(dim=[2,3,4])
    vol_exp  = info['growth_rate'].to(vol_pred.device) * t
    growth   = torch.mean((vol_pred - vol_exp).pow(2))
    return change + GROWTH_PENALTY_WEIGHT * growth

def plot_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for k,v in history.items():
        plt.figure(); plt.plot(v); plt.title(k)
        plt.savefig(os.path.join(save_dir, f"{k}.png"))
        plt.close()


# ---------------------- Training Functions ----------------------
def train_recon(args):
    dl     = DataLoader(Brats2023Dataset(args.train_csv),
                        batch_size=args.bs, shuffle=True)
    val_dl = DataLoader(Brats2023Dataset(args.val_csv),
                        batch_size=args.bs) if args.val_csv else None

    os.makedirs(args.save_dir, exist_ok=True)
    dev = torch.device(args.device)

    enc  = Encoder3D(hidden=args.hidden, emb=args.emb_dim).to(dev)
    dec  = Decoder3D(hidden=args.hidden, emb=args.emb_dim).to(dev)
    vq   = VectorQuantizer(512, args.emb_dim, args.commitment_cost).to(dev)
    disc = Discriminator3D().to(dev)

    opt_g  = optim.Adam(list(enc.parameters())+list(dec.parameters())+list(vq.parameters()), lr=args.lr)
    opt_d  = optim.Adam(disc.parameters(), lr=args.disc_lr, betas=(0.5,0.9))
    scaler = GradScaler()

    writer  = SummaryWriter(args.logdir)
    history = {k:[] for k in (
        'train_recon','train_vq','train_d_real','train_d_fake',
        'val_recon','val_vq','val_d_real','val_d_fake'
    )}

    for ep in range(args.epochs):
        is_pre = ep < args.pretrain_epochs
        phase  = 'Pretrain' if is_pre else 'Adversarial'
        print(f"\n=== {phase} Recon Epoch {ep+1}/{args.epochs} ===")
        sum_r=sum_v=sum_dr=sum_df=0

        for bi,(vol,_,_) in enumerate(dl,1):
            vol = vol.to(dev)
            # discriminator step
            if not is_pre and bi%args.disc_freq==0:
                with autocast():
                    z,_   = vq(enc(vol))
                    fake  = dec(z)
                    pr,pf = disc(vol), disc(fake)
                    loss_d= nn.functional.relu(1-pr).mean() + nn.functional.relu(1+pf).mean()
                opt_d.zero_grad()
                scaler.scale(loss_d).backward()
                scaler.step(opt_d); scaler.update()
                sum_dr+=pr.mean().item(); sum_df+=pf.mean().item()

            # generator + vq step
            with autocast():
                z, vq_loss = vq(enc(vol))
                recon      = dec(z)
                rl         = nn.functional.mse_loss(recon, vol)
                gan_term   = 0.0 if is_pre else -disc(recon).mean()*args.lambda_gan
                loss_g     = rl + vq_loss + gan_term
            opt_g.zero_grad()
            scaler.scale(loss_g).backward()
            scaler.step(opt_g); scaler.update()

            sum_r += rl.item(); sum_v += vq_loss.item()

            if args.batch_verbose and bi%args.batch_verbose==0:
                msg = f"  [B{bi}/{len(dl)}] R={sum_r/bi:.4f} VQ={sum_v/bi:.4f}"
                if not is_pre:
                    c=bi//args.disc_freq
                    msg+=f" D_r={sum_dr/max(1,c):.4f} D_f={sum_df/max(1,c):.4f}"
                print(msg)

        print(f"=> Recon Ep{ep+1}: R={sum_r/len(dl):.4f} VQ={sum_v/len(dl):.4f}"
              +("" if is_pre else f" D_r={sum_dr/(len(dl)//args.disc_freq):.4f} D_f={sum_df/(len(dl)//args.disc_freq):.4f}"))

        if (ep+1)%args.save_ckpt_freq==0:
            torch.save({
                'enc':enc.state_dict(),'dec':dec.state_dict(),
                'vq':vq.state_dict(),'disc':disc.state_dict()
            }, os.path.join(args.save_dir, f"recon_ep{ep+1}.pth"))

    writer.close()


def train_temporal(args):
    dl     = DataLoader(LumiereDataset(args.train_csv),
                        batch_size=args.bs, shuffle=True)
    val_dl = DataLoader(LumiereDataset(args.val_csv),
                        batch_size=args.bs) if args.val_csv else None

    os.makedirs(args.save_dir, exist_ok=True)
    dev = torch.device(args.device)

    enc  = Encoder3D(hidden=args.hidden, emb=args.emb_dim).to(dev)
    dec  = Decoder3D(hidden=args.hidden, emb=args.emb_dim).to(dev)
    vq   = VectorQuantizer(512, args.emb_dim, args.commitment_cost).to(dev)
    disc = Discriminator3D().to(dev)
    func = ODEFunc(args.emb_dim).to(dev)
    odeb = ODEBlock(func).to(dev)

    opt_g  = optim.Adam(
        list(enc.parameters())+list(dec.parameters())+
        list(vq.parameters())+list(func.parameters()),
        lr=args.lr
    )
    opt_d  = optim.Adam(disc.parameters(), lr=args.disc_lr, betas=(0.5,0.9))
    scaler = GradScaler()

    writer  = SummaryWriter(args.logdir)
    history = {k:[] for k in (
        'train_recon','train_vq','train_phys','train_d_real','train_d_fake',
        'val_recon','val_vq','val_phys','val_d_real','val_d_fake'
    )}

    for ep in range(args.epochs):
        is_pre = ep < args.pretrain_epochs
        phase  = 'Pretrain' if is_pre else 'Adversarial'
        print(f"\n=== {phase} Temporal Epoch {ep+1}/{args.epochs} ===")
        sum_r=sum_v=sum_p=sum_dr=sum_df=0

        for bi,(vol,info,tp) in enumerate(dl,1):
            vol, tp = vol.to(dev), tp.to(dev)

            # discriminator step
            if not is_pre and bi%args.disc_freq==0:
                with autocast():
                    z,_  = vq(enc(vol))
                    lat  = []
                    for i in range(z.size(0)):
                        zi   = z[i:i+1]
                        ti   = tp[i].item()
                        tpts = torch.tensor([0.0, ti], device=dev)
                        lat.append( odeb(zi, tpts) )
                    fake = dec(torch.cat(lat,0))
                    pr,pf = disc(vol), disc(fake)
                    loss_d= nn.functional.relu(1-pr).mean() + nn.functional.relu(1+pf).mean()
                opt_d.zero_grad()
                scaler.scale(loss_d).backward()
                scaler.step(opt_d); scaler.update()
                sum_dr+=pr.mean().item(); sum_df+=pf.mean().item()

            # generator + vq + physics step
            with autocast():
                z, vq_loss = vq(enc(vol))
                lat = []
                for i in range(z.size(0)):
                    zi   = z[i:i+1]
                    ti   = tp[i].item()
                    tpts = torch.tensor([0.0, ti], device=dev)
                    lat.append( odeb(zi, tpts) )
                lat   = torch.cat(lat,0)
                recon = dec(lat)

                # auto-crop so recon/vol match exactly
                mh = min(recon.size(2), vol.size(2))
                mw = min(recon.size(3), vol.size(3))
                md = min(recon.size(4), vol.size(4))
                recon = recon[:,:,:mh,:mw,:md]
                vol   = vol[:,:,:mh,:mw,:md]

                rl   = nn.functional.mse_loss(recon, vol)
                pl   = compute_physics_loss(recon, info, tp)
                ganT = 0.0 if is_pre else -disc(recon).mean()*args.lambda_gan
                loss_g = rl + vq_loss + args.physics_weight*pl + ganT

            opt_g.zero_grad()
            scaler.scale(loss_g).backward()
            scaler.step(opt_g); scaler.update()

            sum_r+=rl.item(); sum_v+=vq_loss.item(); sum_p+=pl.item()

            if args.batch_verbose and bi%args.batch_verbose==0:
                msg  = f"  [B{bi}/{len(dl)}] R={sum_r/bi:.4f} VQ={sum_v/bi:.4f} P={sum_p/bi:.4f}"
                if not is_pre:
                    c = bi//args.disc_freq
                    msg+=f" D_r={sum_dr/max(1,c):.4f} D_f={sum_df/max(1,c):.4f}"
                print(msg)

        print(f"=> Temporal Ep{ep+1}: R={sum_r/len(dl):.4f} VQ={sum_v/len(dl):.4f} P={sum_p/len(dl):.4f}"
              +("" if is_pre else f" D_r={sum_dr/(len(dl)//args.disc_freq):.4f} D_f={sum_df/(len(dl)//args.disc_freq):.4f}"))

        if (ep+1)%args.save_ckpt_freq==0:
            torch.save({
                'enc':enc.state_dict(), 'dec':dec.state_dict(),
                'vq':vq.state_dict(),   'disc':disc.state_dict(),
                'odefunc':func.state_dict()
            }, os.path.join(args.save_dir,f"temp_ep{ep+1}.pth"))

    writer.close()


# ---------------------- Main ----------------------
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode',     choices=['train_recon','train_temporal'], default='train_recon')
    p.add_argument('--train_csv', required=True)
    p.add_argument('--val_csv',   default=None)
    p.add_argument('--bs',        type=int,   default=1)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--disc_lr',   type=float, default=1e-3)
    p.add_argument('--disc_freq', type=int,   default=1)
    p.add_argument('--epochs',    type=int,   default=50)
    p.add_argument('--pretrain_epochs', type=int, default=0)
    p.add_argument('--save_ckpt_freq',   type=int, default=1)
    p.add_argument('--lambda_gan',        type=float, default=1.0)
    p.add_argument('--commitment_cost',   type=float, default=0.25)
    p.add_argument('--growth_penalty_weight', type=float, default=0.01)
    p.add_argument('--physics_weight',    type=float, default=0.1)
    p.add_argument('--batch_verbose',     type=int,   default=0)
    p.add_argument('--hidden',   type=int, default=64, help='hidden channels')
    p.add_argument('--emb_dim',  type=int, default=64, help='VQ embedding dim')
    p.add_argument('--device',   type=str, default='cuda')
    p.add_argument('--logdir',   type=str, default='runs/exp')
    p.add_argument('--save_dir', type=str, default='outputs')
    args = p.parse_args()

    GROWTH_PENALTY_WEIGHT = args.growth_penalty_weight
    if args.mode == 'train_recon':
        train_recon(args)
    else:
        train_temporal(args)
