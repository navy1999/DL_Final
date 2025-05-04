#!/usr/bin/env python3
"""
Phase 1: 4-ch VQ-GAN reconstruction on BraTS2023
Phase 2: 4-ch VQ-GAN + ODE temporal model on a longitudinal TCIA set

Usage:

# Phase 1: spatial only
python train_recon_then_temporal.py \
    --phase recon \
    --data_root /path/to/BraTS23 \
    --bs 2 --lr 1e-4 --epochs 50 \
    --device cuda \
    --logdir runs/brats_recon \
    --save_dir outputs/brats_recon \
    --lambda_gan 1.0

# Phase 2: temporal on a TCIA‐style longitudinal CSV
python train_recon_then_temporal.py \
    --phase temporal \
    --train_csv train_tciales.csv \
    --val_csv   val_tciales.csv \
    --bs 2 --lr 1e-4 --epochs 50 \
    --device cuda \
    --logdir runs/tcia_temp \
    --save_dir outputs/tcia_temp \
    --lambda_gan 1.0
"""

import os, glob, csv, argparse
import nibabel as nib
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchdiffeq
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#  Dataset Loaders
# -----------------------------------------------------------------------------

class BraTSDataset(Dataset):
    """4-channel BraTS2023 loader: each case folder must contain *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz"""
    MODALITIES = ['t1','t1ce','t2','flair']
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.cases = sorted(os.listdir(root_dir))
        self.transform = transform

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        vols = []
        for m in self.MODALITIES:
            fn = glob.glob(os.path.join(self.root, case, f"*_{m}.nii.gz"))
            if not fn:
                raise FileNotFoundError(f"Missing {m} in {case}")
            data = nib.load(fn[0]).get_fdata().astype('float32')
            vols.append(data)
        vol = np.stack(vols,0)              # [4,H,W,D]
        vol = torch.from_numpy(vol)         # float32
        if self.transform: vol = self.transform(vol)
        return vol

class LongitudinalMRIDataset(Dataset):
    """
    Expects a CSV with rows:
      patient_id, time_delta, case_root
    where each patient folder under case_root/patient_id/ contains:
      patient_id_t{time_delta}_{modality}.nii.gz
    for modalities in ['t1','t1ce','t2','flair'].
    """
    MODALITIES = ['t1','t1ce','t2','flair']
    def __init__(self, csv_path, transform=None):
        self.entries = []
        with open(csv_path) as f:
            for row in csv.reader(f):
                pid, td, root = row
                self.entries.append((pid, float(td), root))
        self.transform = transform

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        pid, td, root = self.entries[idx]
        vols = []
        for m in self.MODALITIES:
            pattern = os.path.join(root, pid, f"{pid}_t{int(td)}_{m}.nii.gz")
            if not os.path.exists(pattern):
                raise FileNotFoundError(pattern)
            data = nib.load(pattern).get_fdata().astype('float32')
            vols.append(data)
        vol = np.stack(vols,0)
        vol = torch.from_numpy(vol)
        if self.transform: vol = self.transform(vol)
        # we dummy a growth_rate here; you can replace with real metadata per patient
        info = {'growth_rate': torch.tensor(0.05)}
        return vol, info, torch.tensor(td, dtype=torch.float32)

# -----------------------------------------------------------------------------
#  Model Components
# -----------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        B,C,H,W,D = x.shape
        flat = x.permute(0,2,3,4,1).reshape(-1,C)
        dist = (flat.pow(2).sum(1,keepdim=True)
               - 2*flat @ self.embedding.weight.t()
               + self.embedding.weight.pow(2).sum(1))
        idx = torch.argmin(dist,dim=1)
        onehot = torch.zeros(idx.size(0), self.embedding.num_embeddings, device=x.device)
        onehot.scatter_(1, idx.unsqueeze(1), 1)
        quant_flat = onehot @ self.embedding.weight
        quant = quant_flat.view(B,H,W,D,C).permute(0,4,1,2,3)
        e_loss = nn.functional.mse_loss(quant.detach(), x)
        q_loss = nn.functional.mse_loss(quant, x.detach())
        loss = q_loss + self.commitment_cost*e_loss
        quant = x + (quant - x).detach()
        return quant, loss

class Encoder3D(nn.Module):
    def __init__(self, in_ch=4, hidden=128, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch,hidden,4,2,1), nn.ReLU(),
            nn.Conv3d(hidden,hidden,4,2,1), nn.ReLU(),
            nn.Conv3d(hidden,emb,3,1,1)
        )
    def forward(self,x): return self.net(x)

class Decoder3D(nn.Module):
    def __init__(self, emb=64, hidden=128, out_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(emb,hidden,4,2,1), nn.ReLU(),
            nn.ConvTranspose3d(hidden,hidden,4,2,1), nn.ReLU(),
            nn.Conv3d(hidden,out_ch,3,1,1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)

class Discriminator3D(nn.Module):
    def __init__(self, in_ch=4, feats=[64,128,256]):
        super().__init__()
        layers=[]
        p=in_ch
        for f in feats:
            layers += [nn.Conv3d(p,f,4,2,1), nn.LeakyReLU(0.2,inplace=True)]
            p=f
        layers += [nn.Conv3d(p,1,4,1,0)]
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,dim*2), nn.ReLU(),
                                 nn.Linear(dim*2,dim))
    def forward(self,t,z): return self.net(z)

class ODEBlock(nn.Module):
    def __init__(self, func, method='rk4', tol=1e-3):
        super().__init__()
        self.func, self.method, self.tol = func, method, tol
    def forward(self, z, tpts):
        B,C,H,W,D = z.shape
        flat = z.reshape(B,-1)
        out = torchdiffeq.odeint(self.func, flat, tpts.to(flat.device),
                                 method=self.method,
                                 atol=self.tol, rtol=self.tol)
        return out[-1].reshape(B,C,H,W,D)

# -----------------------------------------------------------------------------
#  Physics‐informed loss (optional)
# -----------------------------------------------------------------------------

def physics_loss(pred, info, t):
    # smoothness + volume consistency
    diff = pred - pred.detach()
    change_pen = torch.mean(torch.abs(diff)/(t.view(-1,1,1,1,1)+1e-6))
    voxel_vol=1.0
    vol_pred = pred.sum([2,3,4])*voxel_vol
    vol_exp  = info['growth_rate'].to(vol_pred.device)*t
    growth_pen = torch.mean((vol_pred-vol_exp)**2)
    return change_pen + 0.01*growth_pen

# -----------------------------------------------------------------------------
#  Validation & plotting
# -----------------------------------------------------------------------------

def validate(dl, enc, dec, vq, odeb, disc, device, temporal=False, gan=False):
    enc.eval(); dec.eval(); vq.eval(); odeb.eval(); disc.eval()
    mets = dict(recon=0, vq=0, phys=0, d_r=0, d_f=0); n=0
    with torch.no_grad():
        for batch in dl:
            if temporal:
                vol, info, tp = batch
                tp = tp.to(device)
            else:
                vol = batch
            vol = vol.to(device)
            z = enc(vol); q, vl = vq(z)
            if temporal:
                tpts = torch.stack([torch.zeros_like(tp),tp],0)
                q = odeb(q,tpts)
                pl = physics_loss(dec(q), info, tp)
                mets['phys'] += pl.item()
            recon = dec(q)
            rl = nn.functional.mse_loss(recon, vol)
            mets['recon'] += rl.item()
            mets['vq'] += vl.item()
            if gan:
                mets['d_r'] += disc(vol).mean().item()
                mets['d_f'] += disc(recon).mean().item()
            n += 1
    for k in mets: mets[k] /= n
    enc.train(); dec.train(); vq.train(); odeb.train(); disc.train()
    return mets

def plot_history(h, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for k,v in h.items():
        plt.figure(); plt.plot(v); plt.title(k)
        plt.savefig(os.path.join(save_dir, f"{k}.png"))
        plt.close()

# -----------------------------------------------------------------------------
#  Training
# -----------------------------------------------------------------------------

def train_phase_recon(args):
    ds = BraTSDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    device = torch.device(args.device)

    enc = Encoder3D().to(device)
    dec = Decoder3D().to(device)
    vq  = VectorQuantizer(512,64).to(device)
    disc= Discriminator3D().to(device)

    opt_g = optim.Adam(list(enc.parameters())+list(dec.parameters())+list(vq.parameters()),
                       lr=args.lr)
    opt_d = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5,0.9))

    writer = SummaryWriter(args.logdir)
    H = {'r':[],'vq':[],'dr':[],'df':[]}

    for ep in range(args.epochs):
        sr=sv= sdr=sdf=0
        for vol in dl:
            vol = vol.to(device)
            # D step
            with torch.no_grad():
                z = enc(vol); q,_ = vq(z); fake = dec(q)
            pr = disc(vol); pf = disc(fake)
            ld = torch.relu(1.-pr).mean() + torch.relu(1.+pf).mean()
            opt_d.zero_grad(); ld.backward(); opt_d.step()
            # G step
            z = enc(vol); q,vl = vq(z); recon = dec(q)
            rl = nn.functional.mse_loss(recon,vol)
            lg = rl + vl + args.lambda_gan*(-disc(recon).mean())
            opt_g.zero_grad(); lg.backward(); opt_g.step()
            sr += rl.item(); sv += vl.item()
            sdr+= pr.mean().item(); sdf+= pf.mean().item()
        n=len(dl)
        H['r'].append(sr/n); H['vq'].append(sv/n)
        H['dr'].append(sdr/n);H['df'].append(sdf/n)
        writer.add_scalar('R',sr/n,ep); writer.add_scalar('VQ',sv/n,ep)
        writer.add_scalar('D_real',sdr/n,ep); writer.add_scalar('D_fake',sdf/n,ep)
        print(f"[Recon][Ep{ep}] R={sr/n:.4f} VQ={sv/n:.4f} D_r={sdr/n:.4f} D_f={sdf/n:.4f}")
    writer.close()
    plot_history(H, args.save_dir)

def train_phase_temporal(args):
    tds = LongitudinalMRIDataset(args.train_csv)
    vds = LongitudinalMRIDataset(args.val_csv) if args.val_csv else None
    tdl = DataLoader(tds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    vdl = DataLoader(vds, batch_size=args.bs) if vds else None

    device = torch.device(args.device)
    enc = Encoder3D().to(device)
    dec = Decoder3D().to(device)
    vq  = VectorQuantizer(512,64).to(device)
    disc= Discriminator3D().to(device)

    # latent dim
    with torch.no_grad():
        dummy = torch.zeros(1,64,32,32,32)
        latent_dim = dummy.numel()
    func = ODEFunc(latent_dim).to(device)
    odeb = ODEBlock(func).to(device)

    opt_g = optim.Adam(list(enc.parameters())+
                       list(dec.parameters())+
                       list(vq.parameters())+
                       list(func.parameters()),
                       lr=args.lr)
    opt_d = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5,0.9))

    writer = SummaryWriter(args.logdir)
    H = {'r':[],'vq':[],'ph':[],'dr':[],'df':[]}

    for ep in range(args.epochs):
        sr=sv=sph=sdr=sdf=0
        for vol, info, tp in tdl:
            vol, tp = vol.to(device), tp.to(device)
            # D
            with torch.no_grad():
                z = enc(vol); q,_ = vq(z)
                tpts = torch.stack([torch.zeros_like(tp),tp],0)
                lat = odeb(q,tpts)
                fake = dec(lat)
            pr, pf = disc(vol), disc(fake)
            ld = torch.relu(1.-pr).mean()+torch.relu(1.+pf).mean()
            opt_d.zero_grad(); ld.backward(); opt_d.step()
            # G
            z = enc(vol); q,vl = vq(z)
            tpts = torch.stack([torch.zeros_like(tp),tp],0)
            lat = odeb(q,tpts)
            recon = dec(lat)
            rl = nn.functional.mse_loss(recon,vol)
            pl = physics_loss(recon, info, tp)
            lg = rl + vl + 0.1*pl + args.lambda_gan*(-disc(recon).mean())
            opt_g.zero_grad(); lg.backward(); opt_g.step()
            sr+=rl.item(); sv+=vl.item(); sph+=pl.item()
            sdr+=pr.mean().item(); sdf+=pf.mean().item()

        n=len(tdl)
        H['r'].append(sr/n); H['vq'].append(sv/n); H['ph'].append(sph/n)
        H['dr'].append(sdr/n);H['df'].append(sdf/n)
        writer.add_scalar('R',sr/n,ep); writer.add_scalar('VQ',sv/n,ep)
        writer.add_scalar('Phys',sph/n,ep)
        writer.add_scalar('D_real',sdr/n,ep); writer.add_scalar('D_fake',sdf/n,ep)
        print(f"[Temp][Ep{ep}] R={sr/n:.4f} VQ={sv/n:.4f} Ph={sph/n:.4f} D_r={sdr/n:.4f} D_f={sdf/n:.4f}")

        if vdl:
            m = validate(vdl, enc, dec, vq, odeb, disc, device, temporal=True, gan=True)
            print("  >> val:", m)

    writer.close()
    plot_history(H, args.save_dir)

# -----------------------------------------------------------------------------
#  CLI & dispatch
# -----------------------------------------------------------------------------

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase', choices=['recon','temporal'], required=True)
    # recon args
    p.add_argument('--data_root', help="BraTS23 root for recon phase")
    # temporal args
    p.add_argument('--train_csv', help="CSV for longitudinal TCIA")
    p.add_argument('--val_csv',   default=None)
    # shared
    p.add_argument('--bs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--device', default='cuda')
    p.add_argument('--logdir', default='runs/exp')
    p.add_argument('--save_dir', default='outputs')
    p.add_argument('--lambda_gan', type=float, default=1.0)
    args = p.parse_args()

    if args.phase=='recon':
        assert args.data_root, "--data_root is required for recon"
        train_phase_recon(args)
    else:
        assert args.train_csv, "--train_csv is required for temporal"
        train_phase_temporal(args)
