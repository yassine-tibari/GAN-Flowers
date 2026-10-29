# Jardin GAN — génération de fleurs par DCGAN

Application web (Flask + HTML/CSS/JS) qui génère des images de fleurs avec un **DCGAN**
(PyTorch) entraîné sur le dataset **Flowers102** de torchvision.

## Structure
```
GAN_Flowers/
├── app.py              # backend Flask (API + pages)
├── train.py            # entraînement du DCGAN
├── generate.py         # génération (CLI + utilisé par app.py)
├── models/             # generator.py, discriminator.py, generator.pth (après entraînement)
├── dataset/flowers/    # téléchargé automatiquement par torchvision
├── outputs/training/   # grilles d'échantillons + courbe de pertes
├── outputs/generated/  # images générées depuis l'interface
├── static/{css,js}/
└── templates/index.html
```

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Entraîner le modèle
```bash
python train.py                  # 150 époques par défaut (GPU recommandé)
python train.py --epochs 50      # essai rapide
python train.py --resume         # reprendre un entraînement interrompu
```
Le dataset (~330 Mo) est téléchargé dans `dataset/flowers/`. À chaque époque, les poids sont
enregistrés dans `models/generator.pth` ; une grille d'échantillons est écrite dans
`outputs/training/` toutes les 5 époques (+ `loss_curve.png`).
Ordre de grandeur : ~10 s/époque sur GPU récent, plusieurs minutes/époque sur CPU.

## 2. Lancer l'application
```bash
python app.py        # http://127.0.0.1:5000
```
Choisissez le nombre d'images (1–24), la taille (64–512 px) et éventuellement une graine
(la même graine redonne les mêmes fleurs). Cliquez sur une image pour l'agrandir,
téléchargez-la seule ou toutes en `.zip`.

## Génération en ligne de commande
```bash
python generate.py -n 16 --seed 7 --size 256
```

## API
| Route | Méthode | Description |
|---|---|---|
| `/api/status` | GET | `{model_ready: bool}` |
| `/api/generate` | POST | `{count, size, seed?}` → URLs des images générées |
| `/download/<fichier>` | GET | téléchargement d'une image |
| `/api/zip` | POST | `{files:[...]}` → archive zip |

## Notes
Les images sont natives en 64×64 ; les tailles supérieures sont un agrandissement (Lanczos).
La qualité dépend du nombre d'époques : attendez-vous à des formes/couleurs florales
convaincantes mais floues, typiques d'un DCGAN 64×64 sur ~8 000 images.
