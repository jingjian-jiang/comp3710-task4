#!/usr/bin/env python3
# Task 4 - Auto VAE pipeline for MRI
# If dataset.npy not found, it will preprocess from data_dir (NIfTI files)
import os, argparse, glob
import numpy as np
import nibabel as nib
from skimage.transform import resize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

def load_nii(path):
    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def make_dataset(data_dir, out="dataset.npy", img_size=64, slices=16):
    files = sorted(glob.glob(os.path.join(data_dir, "*.nii*")))
    if not files:
        print(f"No NIfTI files in {data_dir}, fallback to synthetic data")
        return None
    all_imgs = []
    for fp in files:
        vol = load_nii(fp)
        d = vol.shape[2]
        start = max(0, d//2 - slices//2)
        end = min(d, start + slices)
        for k in range(start, end):
            slc = vol[:,:,k]
            slc = resize(slc, (img_size, img_size), anti_aliasing=True)
            all_imgs.append(slc[None, ...])
    arr = np.stack(all_imgs, axis=0).astype(np.float32)
    np.save(out, arr)
    print(f"Saved {arr.shape} to {out}")
    return out

class NpyMRIDataset(torch.utils.data.Dataset):
    def __init__(self, path=None, n_synth=512, img_size=64):
        if path and os.path.isfile(path):
            self.data = np.load(path).astype(np.float32)
            print(f"Loaded dataset: {self.data.shape}")
        else:
            print("⚠️ Using synthetic data")
            self.data = np.random.rand(n_synth,1,img_size,img_size).astype(np.float32)
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, idx): return self.data[idx]

class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.ReLU()
        )
        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8, latent_dim)
        self.fc_up = nn.Linear(latent_dim, 128*8*8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Sigmoid()
        )
    def encode(self,x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        return mu + torch.randn_like(std)*std
    def decode(self,z):
        h = self.fc_up(z).view(-1,128,8,8)
        return self.dec(h)
    def forward(self,x):
        mu,logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon,x,mu,logvar):
    bce = nn.functional.binary_cross_entropy(recon,x,reduction="sum")
    kld = -0.5*torch.sum(1+logvar - mu.pow(2)-logvar.exp())
    return (bce+kld)/x.size(0)

@torch.no_grad()
def save_recon(model,batch,outdir):
    x = batch.to(next(model.parameters()).device)
    recon,_,_ = model(x)
    x = x.cpu().numpy(); r = recon.cpu().numpy()
    strip = np.concatenate([np.concatenate([x[i,0],r[i,0]],axis=1) for i in range(min(8,len(x)))],axis=0)
    plt.imshow(strip,cmap="gray"); plt.axis("off")
    plt.savefig(os.path.join(outdir,"recon_vs_input.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",type=str,default="Pictures")
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--outdir",type=str,default="results_task4")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dataset_path = "dataset.npy"
    if not os.path.isfile(dataset_path):
        make_dataset(args.data_dir, out=dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NpyMRIDataset(path=dataset_path)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for ep in range(1,args.epochs+1):
        model.train(); total=0
        for batch in dl:
            batch = batch.to(device)
            opt.zero_grad()
            recon,mu,logvar = model(batch)
            loss = vae_loss(recon,batch,mu,logvar)
            loss.backward(); opt.step()
            total += loss.item()*batch.size(0)
        avg = total/len(ds)
        losses.append(avg)
        print(f"Epoch {ep}/{args.epochs} | loss={avg:.4f}")

    plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig(os.path.join(args.outdir,"loss_curve.png"))

    with torch.no_grad():
        z = torch.randn(16,16).to(device)
        samples = model.decode(z).cpu().numpy()
        grid = np.concatenate([samples[i,0] for i in range(16)],axis=1)
        plt.imshow(grid,cmap="gray"); plt.axis("off")
        plt.savefig(os.path.join(args.outdir,"vae_samples.png"))
    for batch in dl:
        save_recon(model,batch,args.outdir)
        break
    torch.save(model.state_dict(), os.path.join(args.outdir,"vae_mri.pt"))
    print("Done. Results in", args.outdir)

if __name__ == "__main__":
    main()
