# Roadwatch web

Static, browser-based Roadwatch demo for deployment to Vercel. Camera access,
face landmarks, eye-closure checks, and spoken alerts run in the visitor's
browser. Camera frames are not uploaded or recorded.

## Run locally

From this directory, start any static HTTP server, for example:

```sh
python3 -m http.server 8000
```

Open <http://localhost:8000>. Browsers permit camera access on `localhost`;
deployed use requires HTTPS.

## Deploy to Vercel with GitHub

1. Create a GitHub repository for this `roadwatch-web` directory.
2. From this directory, commit and push these static files:

   ```sh
   git init
   git add index.html app.js style.css vercel.json README.md
   git commit -m "Create Roadwatch browser app"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/roadwatch-web.git
   git push -u origin main
   ```

   Replace `YOUR-USERNAME` with your GitHub username and create that empty
   repository on GitHub first. If Git asks, configure your author name/email.
3. Sign in at <https://vercel.com>, choose **Add New → Project**, and import
   the `roadwatch-web` GitHub repository.
4. Keep the project root at the repository root. Select **Other** as the
   framework preset if Vercel does not detect a framework. Leave the build and
   install commands empty; this is a static HTML/CSS/JavaScript site.
5. Choose **Deploy**. After the build completes, open the generated
   `https://…vercel.app` address.
6. Visit the deployed page, press **Start monitoring**, and allow camera access.
   The browser must be using HTTPS; camera permission cannot be granted on an
   ordinary remote HTTP URL.
7. To update the site, push later changes to `main`; Vercel automatically
   creates a new deployment. Add a custom domain under the project's
   **Settings → Domains** if desired.

The first monitoring session needs internet access to load the pinned
MediaPipe JavaScript, WebAssembly runtime, and face-landmarker model. Inference
after loading is performed in the browser. These external assets are served
by jsDelivr and Google Storage, not by the webcam upload path (which does not
exist in this app).

## Safety

This is an experimental demo, not a certified driver-safety system. Do not
interact with setup, permissions, or the screen while driving. Eye closure can
be misdetected due to lighting, camera angle, glasses, and model limitations.
