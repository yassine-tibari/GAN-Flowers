"""Backend Flask : génération, visualisation et téléchargement de fleurs DCGAN."""
import io
import re
import threading
import zipfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory

from generate import OUT_DIR, WEIGHTS, generate_images, load_generator, save_images

app = Flask(__name__)

MAX_IMAGES = 24
ALLOWED_SIZES = (64, 128, 256, 512)
MAX_KEPT_FILES = 300  # on purge les plus anciens au-delà
FILENAME_RE = re.compile(r"^fleur_[0-9a-f]{12}\.png$")

_generator = None
_lock = threading.Lock()


def get_generator():
    global _generator
    with _lock:
        if _generator is None:
            _generator = load_generator()
        return _generator


def purge_old_files():
    files = sorted(OUT_DIR.glob("fleur_*.png"), key=lambda p: p.stat().st_mtime)
    for p in files[:-MAX_KEPT_FILES]:
        p.unlink(missing_ok=True)


@app.get("/")
def index():
    return render_template("index.html", max_images=MAX_IMAGES, sizes=ALLOWED_SIZES)


@app.get("/api/status")
def status():
    return jsonify(model_ready=WEIGHTS.exists())


@app.post("/api/generate")
def api_generate():
    data = request.get_json(silent=True) or {}
    try:
        count = int(data.get("count", 8))
        size = int(data.get("size", 256))
        seed = data.get("seed")
        seed = None if seed in (None, "") else int(seed)
    except (TypeError, ValueError):
        return jsonify(error="Paramètres invalides."), 400
    if not 1 <= count <= MAX_IMAGES:
        return jsonify(error=f"Le nombre d'images doit être entre 1 et {MAX_IMAGES}."), 400
    if size not in ALLOWED_SIZES:
        return jsonify(error=f"Taille autorisée : {', '.join(map(str, ALLOWED_SIZES))}."), 400
    if seed is not None and not 0 <= seed < 2**31 - 64:
        return jsonify(error="La graine doit être un entier entre 0 et 2147483583."), 400

    try:
        net = get_generator()
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503

    images, used_seed = generate_images(net, count, seed, size)
    names = save_images(images)
    purge_old_files()
    return jsonify(
        seed=used_seed,
        size=size,
        images=[{"name": n,
                 "url": f"/generated/{n}",
                 "download": f"/download/{n}",
                 "seed": used_seed + i} for i, n in enumerate(names)],
    )


@app.get("/generated/<path:name>")
def serve_generated(name):
    if not FILENAME_RE.match(name):
        return jsonify(error="Fichier invalide."), 404
    return send_from_directory(OUT_DIR, name, max_age=3600)


@app.get("/download/<path:name>")
def download(name):
    if not FILENAME_RE.match(name):
        return jsonify(error="Fichier invalide."), 404
    return send_from_directory(OUT_DIR, name, as_attachment=True)


@app.post("/api/zip")
def download_zip():
    names = (request.get_json(silent=True) or {}).get("files", [])
    if not isinstance(names, list) or not names or len(names) > MAX_IMAGES:
        return jsonify(error="Liste de fichiers invalide."), 400
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            if isinstance(n, str) and FILENAME_RE.match(n) and (OUT_DIR / n).is_file():
                z.write(OUT_DIR / n, arcname=n)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="fleurs_gan.zip")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
