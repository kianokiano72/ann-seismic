# ANN — Seismic Period Estimator (Master 2 Génie Civil, UAMOB)

A static, self-contained website that runs the **real trained neural network** (V7 ensemble of 5 seeds)
directly in the browser to estimate the fundamental periods **Tₓ / Tᵧ** of RC buildings in Algeria.

## Files (this is all you deploy)

```
website/
├── index.html       # page + styling
├── app.js           # real ANN prediction, charts, predictor, QR
└── model-data.js    # exported weights + scalers + 96-building dataset
```

No build step. No server. Everything is plain HTML/JS. External libraries
(Tailwind, Chart.js, QR generator, Google Fonts) load from a CDN, so the site
needs an internet connection the first time it opens.

---

## Deploy — pick ONE option

### Option A — Netlify Drop (easiest, ~30 seconds, no account login needed to test)

1. Go to **https://app.netlify.com/drop**
2. Drag the whole **`website`** folder onto the page.
3. You instantly get a public URL like `https://random-name.netlify.app`.
4. (Optional) Create a free account to keep it and rename it.

### Option B — GitHub Pages (best for a permanent academic link)

1. Create a free GitHub account, then a new **public** repository, e.g. `ann-seismic`.
2. Upload the **contents** of the `website/` folder (so `index.html` is at the repo root).
   - Web way: repo → **Add file → Upload files** → drag `index.html`, `app.js`, `model-data.js` → **Commit**.
   - CLI way:
     ```bash
     cd website
     git init
     git add index.html app.js model-data.js README.md
     git commit -m "ANN seismic period estimator website"
     git branch -M main
     git remote add origin https://github.com/USERNAME/ann-seismic.git
     git push -u origin main
     ```
3. Repo → **Settings → Pages** → *Build and deployment* → Source: **Deploy from a branch** →
   Branch: **main**, folder: **/ (root)** → **Save**.
4. Wait ~1 minute. Your site is at:
   ```
   https://USERNAME.github.io/ann-seismic/
   ```

### Option C — Vercel

1. Sign in at **https://vercel.com** with GitHub.
2. **Add New → Project** → import the repo → framework preset **Other** → **Deploy**.
3. You get `https://ann-seismic.vercel.app`.

---

## After deploying: fix the QR code link (optional)

The QR code **auto-detects** the address once the site is online — when opened over
`https://…`, it encodes the current page URL. Nothing to do in most cases.

If you want the QR to show a fixed address even when opened locally, edit one line in
`app.js`:

```js
const DEPLOYED_URL = "https://USERNAME.github.io/ann-seismic/";
```

---

## Run locally

Just double-click `index.html`. (The QR will point to `DEPLOYED_URL` until the site is hosted.)

## Credits

DJOUABI Youcef Mohammed Elamine · YOUSFI Houssam — Supervisor: Dr. AOUARI Issam
Akli Mohand Oulhadj University — Bouira · Génie Civil · 2025/2026
