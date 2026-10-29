"""Entraînement d'un DCGAN sur Flowers102 (torchvision).

Usage :
    python train.py                       # 150 époques, téléchargement auto du dataset
    python train.py --epochs 50 --batch-size 64
    python train.py --resume              # reprend depuis models/checkpoint.pth
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import Flowers102

from models import IMAGE_SIZE, NZ, weights_init
from models.discriminator import Discriminator
from models.generator import Generator

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset" / "flowers"
MODELS_DIR = ROOT / "models"
TRAIN_OUT = ROOT / "outputs" / "training"


def get_dataloader(batch_size: int, workers: int) -> DataLoader:
    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1] (Tanh)
    ])
    # On exploite les 3 splits (8189 images) : pas besoin d'évaluation en GAN.
    parts = [Flowers102(str(DATA_DIR), split=s, transform=transform, download=True)
             for s in ("train", "val", "test")]
    ds = ConcatDataset(parts)
    print(f"Dataset : {len(ds)} images de fleurs")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                      drop_last=True, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


def save_loss_plot(g_losses, d_losses):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(g_losses, label="Générateur")
    plt.plot(d_losses, label="Discriminateur")
    plt.xlabel("Époque"); plt.ylabel("Perte"); plt.legend(); plt.tight_layout()
    plt.savefig(TRAIN_OUT / "loss_curve.png", dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sample-every", type=int, default=5, help="sauvegarde une grille toutes les N époques")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    loader = get_dataloader(args.batch_size, args.workers)

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

    start_epoch, g_hist, d_hist = 0, [], []
    ckpt_path = MODELS_DIR / "checkpoint.pth"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        netG.load_state_dict(ck["G"]); netD.load_state_dict(ck["D"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        start_epoch, g_hist, d_hist = ck["epoch"], ck["g_hist"], ck["d_hist"]
        print(f"Reprise à l'époque {start_epoch}")

    real_label, fake_label = 0.9, 0.0  # label smoothing sur les vraies images

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        g_sum = d_sum = 0.0
        for real, _ in loader:
            real = real.to(device, non_blocking=True)
            b = real.size(0)

            # 1) Discriminateur : maximiser log D(x) + log(1 - D(G(z)))
            netD.zero_grad()
            out_real = netD(real)
            loss_real = criterion(out_real, torch.full((b,), real_label, device=device))
            noise = torch.randn(b, NZ, 1, 1, device=device)
            fake = netG(noise)
            out_fake = netD(fake.detach())
            loss_fake = criterion(out_fake, torch.full((b,), fake_label, device=device))
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optD.step()

            # 2) Générateur : maximiser log D(G(z))
            netG.zero_grad()
            loss_g = criterion(netD(fake), torch.ones(b, device=device))
            loss_g.backward()
            optG.step()

            g_sum += loss_g.item(); d_sum += loss_d.item()

        n = len(loader)
        g_hist.append(g_sum / n); d_hist.append(d_sum / n)
        print(f"[{epoch + 1:>3}/{args.epochs}] loss_D={d_hist[-1]:.3f} loss_G={g_hist[-1]:.3f} "
              f"({time.time() - t0:.0f}s)")

        # Sauvegardes
        torch.save(netG.state_dict(), MODELS_DIR / "generator.pth")
        torch.save({"G": netG.state_dict(), "D": netD.state_dict(),
                    "optG": optG.state_dict(), "optD": optD.state_dict(),
                    "epoch": epoch + 1, "g_hist": g_hist, "d_hist": d_hist}, ckpt_path)
        if (epoch + 1) % args.sample_every == 0 or epoch + 1 == args.epochs:
            netG.eval()
            with torch.no_grad():
                samples = netG(fixed_noise).cpu()
            netG.train()
            vutils.save_image(samples, TRAIN_OUT / f"epoch_{epoch + 1:03d}.png",
                              nrow=8, normalize=True, value_range=(-1, 1), padding=2)
            save_loss_plot(g_hist, d_hist)

    print("Entraînement terminé -> models/generator.pth")


if __name__ == "__main__":
    main()
