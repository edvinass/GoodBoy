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

`public/goodboy.tar.gz` and `install.sh` (in this directory) are release artifacts committed for Railway — Railway only deploys `web/`, not the repo root. Edit the installer at the repo root, then run the release script when you want to cut a new bundle (it auto-bumps the patch version):

```bash
npm run release            # or: ./scripts/release-web-tar.sh
```

Optional versioned copy: `./scripts/release-web-tar.sh --version 0.1.0`

`npm run dev` and `npm run build` do **not** run the release script — `prebuild` only verifies that `public/goodboy.tar.gz` and `install.sh` exist so Railway builds fail loudly if the bundle is missing.

## Custom domain

Point DNS at Railway, then set the domain in the Railway dashboard. The site picks up `window.location.origin` automatically — no env vars required.
