# GoodBoy website

Vue landing page for Railway. Serves `install.sh` so users can install with:

```bash
curl -fsSL https://YOUR-RAILWAY-DOMAIN/install.sh | bash
```

## Local development

```bash
cd web
npm install
npm run dev
```

Open http://localhost:5173 — the install command uses your local origin.

## Deploy to Railway

1. Create a new project on [Railway](https://railway.app).
2. Connect this repo (or deploy from the `web` directory).
3. Set the **root directory** to `web` in service settings.
4. Railway runs `npm run build` then `npm start` (see `railway.toml`).
5. Add a public domain under **Settings → Networking → Generate domain**.

The install URL is always:

```text
https://<your-domain>/install.sh
```

`public/install.sh` is committed for Railway (web-only deploys). Local builds copy from the repo root when `../install.sh` exists. After changing the root `install.sh`, run `cp ../install.sh public/install.sh` and commit, then redeploy.

## Custom domain

Point DNS at Railway, then set the domain in the Railway dashboard. The site picks up `window.location.origin` automatically — no env vars required.
