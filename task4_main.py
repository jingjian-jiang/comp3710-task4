import os, glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ===== Dataset Loader for PNG =====
class BrainPNGDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = glob.glob(os.path.join(folder, "*.png"))
        if len(self.files) == 0:
            raise RuntimeError(f"❌ 没有找到PNG图片，请检查路径: {folder}")
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")  # 灰度模式
        if self.transform:
            img = self.transform(img)
        return img

# ===== VAE 网络结构 =====
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64*16*16, latent_dim)
        self.fc_logvar = nn.Linear(64*16*16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64*16*16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64,16,16)),
            nn.ConvTranspose2d(64,32,4,stride=2,padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,stride=2,padding=1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        h_dec = self.fc_decode(z)
        x_recon = self.decoder(h_dec)
        return x_recon, mu, logvar

# ===== VAE Loss =====
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + KLD) / x.size(0)

# ===== Main Training =====
def main():
    data_dir = "Pictures"   # PNG 图片所在目录
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    dataset = BrainPNGDataset(data_dir, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        for imgs in loader:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), "vae_png.pt")
    print("✅ 模型已保存到 vae_png.pt")

if __name__ == "__main__":
    main()
