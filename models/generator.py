import torch.nn as nn
from . import NZ, NGF, NC


class Generator(nn.Module):
    """DCGAN : vecteur latent (N, NZ, 1, 1) -> image (N, 3, 64, 64) dans [-1, 1]."""

    def __init__(self, nz: int = NZ, ngf: int = NGF, nc: int = NC):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)
