"""Génération d'images avec le générateur entraîné (utilisable en CLI et par app.py).

Usage :
    python generate.py -n 16 --seed 7 --size 256
"""
import argparse
import secrets
import uuid
from pathlib import Path

import torch
from PIL import Image

from models import NZ
from models.generator import Generator

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "generator.pth"
OUT_DIR = ROOT / "outputs" / "generated"


def load_generator(weights: Path = WEIGHTS, device: str = "cpu") -> Generator:
    if not weights.exists():
        raise FileNotFoundError(
            f"Poids introuvables : {weights}. Lancez d'abord `python train.py`.")
    g = Generator().to(device)
    g.load_state_dict(torch.load(weights, map_location=device))
    g.eval()
    return g


@torch.no_grad()
def generate_images(netG: Generator, n: int, seed: int | None = None,
                    size: int = 256, device: str = "cpu") -> tuple[list[Image.Image], int]:
    """Retourne (images PIL, seed utilisée). Image i = seed + i -> reproductible."""
    if seed is None:
        seed = secrets.randbelow(2**31 - 64)
    images = []
    for i in range(n):
        gen = torch.Generator().manual_seed(seed + i)
        z = torch.randn(1, NZ, 1, 1, generator=gen).to(device)
        x = netG(z)[0].cpu().clamp(-1, 1)
        arr = ((x + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
        img = Image.fromarray(arr, "RGB")
        if size != img.width:
            img = img.resize((size, size), Image.LANCZOS)
        images.append(img)
    return images, seed


def save_images(images: list[Image.Image], out_dir: Path = OUT_DIR) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for img in images:
        name = f"fleur_{uuid.uuid4().hex[:12]}.png"
        img.save(out_dir / name, format="PNG")
        names.append(name)
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    net = load_generator()
    imgs, used = generate_images(net, a.n, a.seed, a.size)
    print(f"seed={used}")
    for f in save_images(imgs):
        print(OUT_DIR / f)
