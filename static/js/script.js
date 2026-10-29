(() => {
  const $ = (s) => document.querySelector(s);
  const form = $('#gen-form'), btn = $('#btn-generate'), grid = $('#grid');
  const results = $('#results'), empty = $('#empty'), msg = $('#message'), info = $('#results-info');
  const count = $('#count'), countOut = $('#count-out');
  const lightbox = $('#lightbox'), lbImg = $('#lightbox img'), lbDl = $('#lb-download');
  let currentFiles = [];

  count.addEventListener('input', () => (countOut.textContent = count.value));

  function showMessage(text) {
    msg.textContent = text;
    msg.hidden = !text;
  }

  function render(data) {
    grid.replaceChildren();
    currentFiles = data.images.map((i) => i.name);
    data.images.forEach((img, idx) => {
      const card = document.createElement('article');
      card.className = 'card';
      card.style.animationDelay = `${idx * 40}ms`;

      const thumb = document.createElement('button');
      thumb.type = 'button';
      thumb.className = 'thumb';
      thumb.setAttribute('aria-label', `Agrandir la fleur ${idx + 1}`);
      const el = document.createElement('img');
      el.src = img.url; el.alt = `Fleur générée ${idx + 1}`; el.loading = 'lazy';
      thumb.appendChild(el);
      thumb.addEventListener('click', () => openLightbox(img));

      const meta = document.createElement('div');
      meta.className = 'meta';
      const s = document.createElement('span');
      s.textContent = `graine ${img.seed}`;
      const a = document.createElement('a');
      a.href = img.download; a.textContent = 'Télécharger';
      meta.append(s, a);

      card.append(thumb, meta);
      grid.appendChild(card);
    });
    info.textContent = `${data.images.length} image(s) · ${data.size}×${data.size} px · graine de départ ${data.seed}`;
    results.hidden = false;
    empty.hidden = true;
  }

  function openLightbox(img) {
    lbImg.src = img.url;
    lbDl.href = img.download;
    lightbox.hidden = false;
  }
  lightbox.addEventListener('click', (e) => { if (e.target !== lbDl) lightbox.hidden = true; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') lightbox.hidden = true; });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    showMessage('');
    btn.disabled = true; btn.classList.add('loading');
    btn.querySelector('.label').textContent = 'Génération…';
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: count.value, size: $('#size').value, seed: $('#seed').value }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `Erreur serveur (${res.status})`);
      render(data);
    } catch (err) {
      showMessage(err.message || 'Impossible de contacter le serveur.');
    } finally {
      btn.disabled = false; btn.classList.remove('loading');
      btn.querySelector('.label').textContent = 'Générer';
    }
  });

  $('#btn-zip').addEventListener('click', async () => {
    if (!currentFiles.length) return;
    try {
      const res = await fetch('/api/zip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files: currentFiles }),
      });
      if (!res.ok) throw new Error('Échec de la création du zip.');
      const url = URL.createObjectURL(await res.blob());
      const a = Object.assign(document.createElement('a'), { href: url, download: 'fleurs_gan.zip' });
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (err) { showMessage(err.message); }
  });

  fetch('/api/status').then((r) => r.json()).then((s) => {
    if (!s.model_ready) showMessage("Modèle non entraîné : lancez d'abord « python train.py » (voir README).");
  }).catch(() => {});
})();
