"""Hyperparamètres d'architecture partagés entre entraînement et génération."""
import torch.nn as nn

NZ = 100        # dimension du vecteur latent
NGF = 64        # largeur de base du générateur
NDF = 64        # largeur de base du discriminateur
NC = 3          # canaux de l'image (RGB)
IMAGE_SIZE = 64  # résolution native des images générées


def weights_init(m: nn.Module) -> None:
    """Initialisation recommandée par le papier DCGAN : N(0, 0.02)."""
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
