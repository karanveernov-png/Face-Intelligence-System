# Frontend

React + Vite + TypeScript dashboard for the Face Intelligence backend.

## Setup (Windows PowerShell)

```powershell
cd face-intelligence\frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The dev server proxies `/api`, `/health`, and
`/ws` to the backend at `127.0.0.1:8000` (see `vite.config.ts`) — start the
backend first (see the top-level `README.md`).

`npm run build` produces a static production bundle in `dist/` (verified
during this build: `tsc -b && vite build` completes with zero type errors —
see `CHANGELOG.md`).

## What's here

- The model status check is manual: click **Test models** in the top bar to
  run the backend's model checks. Opening the dashboard does not load or
  warm the models.
- Age estimation starts disabled. The analysis-task switches control the
  selected tasks, and changes made during webcam analysis apply to the next
  frame.
- **Analyze tab**: upload an image or video, see it rendered with real
  bounding boxes, landmarks, and per-face results drawn directly from the
  API response — nothing illustrative or placeholder.
- **Webcam tab**: live analysis over a WebSocket (`/ws/analyze`). Camera
  access is only requested when you click "Start webcam" (the browser's own
  permission prompt), and a pulsing "LIVE" badge is visible the entire time
  frames are being sent — see `components/WebcamPanel.tsx`.
- **Enroll tab**: name + photo, gated by an explicit consent checkbox
  (`components/ConsentDialog.tsx`) that must be ticked before the API call
  fires — this is the actual consent-collection step the backend's
  `consent_confirmed` flag assumes already happened (see
  `../docs/PRIVACY.md`).
- Every panel surfaces the backend's uncertainty language as-is (match
  distance, "uncertain" expression state, age's approximate-estimate note)
  rather than reducing results to bare numbers.

## Known issue (documented, not silently ignored)

`npm audit` reports one moderate and one high severity finding in `esbuild`
(bundled by this Vite version) — `GHSA-67mh-4wv8-2f99`, which allows any
website to send requests to the **dev server** (not the production build)
and read the response, while `npm run dev` is running. A fix is available
via a major Vite version bump (`npm audit fix --force` → Vite 8), which was
deliberately **not** applied here without validating it doesn't break the
build — that's a real risk to weigh, not a one-line fix to apply blindly.
If you want to take the upgrade, run it yourself and re-run `npm run build`
to confirm nothing broke, or wait for a non-breaking patch release.

## Not built yet

- No test suite for the frontend itself (the backend's 39 tests cover the
  API contract this UI depends on; the UI code itself was verified by a
  real, passing production build and a live proxy round-trip to the
  backend, not by component tests — see `CHANGELOG.md`).
- No visual regression/screenshot testing.
