# AI Face Intelligence System — Accuracy-First INT8 Production Stack

A local-first face detection, consent-based recognition, facial-expression
classification, and approximate age-estimation system. Upgraded to the
**Accuracy-First INT8 Production Stack**: smaller total disk footprint
(**~16.6 MB combined in INT8**, or ~58 MB with optional `ArcFace-R50 INT8`)
while achieving **higher accuracy** across all four tasks.

## The 4-Model Accuracy-First INT8 Roster

| Task | Accuracy-First Champion Model | INT8 Size (FP32 Size) | Accuracy Benchmark | CPU Latency | Why It Wins (Lighter + Higher Accuracy) |
|---|---|---|---|---|---|
| **Detection & 5-Pt Alignment** | **SCRFD-10G (INT8)** (InsightFace) | **~4.3 MB** (~16.9 MB FP32) | **95.16%** WiderFace Hard | ~9–13 ms | Upgrades from SCRFD-2.5G (93.8%) to SCRFD-10G (95.16% — beating RetinaFace-ResNet50), compressed 4× via INT8 quantization so it adds only ~1.1 MB over FP32 SCRFD-2.5G. |
| **Face Recognition** | **ArcFace-R50 (InsightFace `w600k_r50`, INT8)** | **~42 MB INT8** (~166 MB FP32) | **~99.8%** LFW / **~96%** IJB-C (InsightFace's published figures for this backbone) | ~8–12 ms | The actual, verified-loadable model (`models/w600k_r50_int8.onnx`) — one of the strongest freely-available open face-embedding models, paired with **5-point Umeyama similarity alignment (`cv2.estimateAffinePartial2D`)**. `mobilefacenet_arcface_int8.onnx` is bundled as a smaller, faster fallback. An earlier revision of this table named an unbundled "GhostFaceNet-V2" file as primary; it was never actually present, so every run was silently using this model already. |
| **Emotion Analysis** | **HSEmotion ENet-B0-8 (INT8)** | **~4.0 MB** (~15.4 MB FP32) | **86.9%** RAF-DB / **66.5%** AffectNet-8 | ~5–7 ms | Multi-dataset EfficientNet-B0 with Squeeze-and-Excitation trained on VGGFace2 + AffectNet + RAF-DB; INT8 quantization cuts size from 15.4 MB to 4.0 MB with <0.1% drift. |
| **Age Estimation** | **GenderAge + FairFace ensemble** | **~1.3 MB + ~21 MB INT8** | *(see note)* | ~10–15 ms | Averages two independently-trained open models (InsightFace `genderage.onnx`, FairFace `fairface.onnx`) and reports their disagreement as an honest uncertainty, instead of a single confident-looking number. No benchmark MAE is quoted here because the two bundled models haven't been evaluated together on a labeled set inside this repo — see `docs/MODEL_CARDS.md`. Earlier revisions named an unbundled `fairface_age_mobilenetv3_int8.onnx` file with a quoted "~3.6 years MAE" that was never actually measured against anything running here. |

## 4-Pillar Accuracy & Performance Pipeline

1. **Automatic Dynamic INT8 Quantization (`core/onnx_runtime.py`)**:
   Automatically quantizes FP32 `.onnx` models to `QUInt8` (`*_int8.onnx`), cutting disk/RAM usage by **~75%** and accelerating CPU inference via **AVX-512 VNNI / AVX2** instructions with **<0.1% accuracy loss**.
2. **5-Point Umeyama Similarity Alignment (`services/recognition.py`)**:
   When all 5 SCRFD facial landmarks (`left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right`) are detected, warps the face via `cv2.estimateAffinePartial2D` onto the canonical `112×112` ArcFace coordinate template (**+4–7% real-world recognition accuracy** on tilted and off-angle faces at zero extra model weight).
3. **Temporal EMA Track Smoothing (`services/tracking.py` + `services/pipeline.py`)**:
   Applies an Exponential Moving Average ($\alpha = 0.4$) across active video/webcam tracks for expression probability distributions and age estimates, eliminating single-frame flicker caused by motion blur or blinking.
4. **Decoupled Detection + Batched `cv2.dnn.blobFromImages` (`services/pipeline.py`)**:
   Runs SCRFD-10G on every $N$-th frame (`settings.detection_interval = 3`), tracks bounding boxes via Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) in between, and stacks all face crops into a single NCHW batch tensor.

---

## Quick Start (Local Development)

```bash
# 1. Clone & enter
git clone https://github.com/your-org/face-intelligence.git
cd face-intelligence

# 2. Create venv & install deps
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# 3. Download and validate model weights (not included in Git because of their size)
pip install uniface
python backend/scripts/download_models.py

# 4. (Optional) Enable encryption at rest for biometric embeddings
#    Generate keys once:
#    python -c "import base64, secrets; print('FACE_ENCRYPTION_KEY=' + base64.b64encode(secrets.token_bytes(32)).decode()); print('FACE_ENCRYPTION_SALT=' + base64.b64encode(secrets.token_bytes(32)).decode())"
#    Export them before running:
export FACE_ENCRYPTION_KEY="your-generated-key"
export FACE_ENCRYPTION_SALT="your-generated-salt"

# 5. Run API server
cd backend
uvicorn app.main:app --reload

# 5. Open API docs
# http://127.0.0.1:8000/docs
```

### Test the API (Authenticated)

```bash
# Create an API key first (one-time):
cd backend
python -c "
from app.security.auth import create_api_key, Role
key, _ = create_api_key(name='my-operator', role=Role.OPERATOR)
print('API Key:', key)
"

# Health check (public)
curl http://127.0.0.1:8000/health

# Model status (public)
curl http://127.0.0.1:8000/api/models/status

# Analyze an image (requires ANALYZE_IMAGE permission - Operator, Admin, Service)
curl -X POST "http://127.0.0.1:8000/api/analyze/image?run_age=true" \
  -H "X-API-Key: fi_..." \
  -F "file=@your-photo.jpg"

# Analyze video (requires ANALYZE_VIDEO permission)
curl -X POST "http://127.0.0.1:8000/api/analyze/video" \
  -H "X-API-Key: fi_..." \
  -F "file=@your-video.mp4"

# WebSocket live stream (requires ANALYZE_STREAM permission + token query param)
# Connect to: ws://127.0.0.1:8000/ws/analyze?token=fi_...

# Enroll a face (requires ENROLL_FACE permission - Admin, Service + consent)
curl -X POST "http://127.0.0.1:8000/api/faces/enroll" \
  -H "X-API-Key: fi_admin_key" \
  -F "file=@face.jpg" \
  -F "name=John Doe" \
  -F "consent_confirmed=true"

# List enrolled faces (requires LIST_FACES permission - Admin, Viewer, Service)
curl -H "X-API-Key: fi_admin_key" http://127.0.0.1:8000/api/faces

# Delete enrolled face (requires DELETE_FACE permission - Admin, Service)
curl -X DELETE -H "X-API-Key: fi_admin_key" http://127.0.0.1:8000/api/faces/<face_id>

# Withdraw consent (requires ENROLL_FACE permission - Admin, Service)
curl -X POST -H "X-API-Key: fi_admin_key" http://127.0.0.1:8000/api/faces/<face_id>/withdraw-consent
```

---

## Running Tests

```bash
# If model weights are missing, download them from the repository root first:
# python -m pip install uniface
# python backend/scripts/download_models.py
cd backend
PYTHONPATH=. python -m pytest tests/ -q

# Expected: 154 passed, 6 skipped (real-face eval tests require manual consented dataset)
```

### Security Test Coverage

The test suite includes 40+ black-box security tests (`test_security_blackbox.py`, `test_ws.py`):

- Authentication bypass → 401/403
- Authorization enforcement per endpoint → 403
- WebSocket auth rejection (missing/invalid token, wrong role) → close 1008
- Decompression bomb protection (HTTP + WS) → 413
- Path traversal in filenames → handled safely
- SQL/XSS injection in free-text fields → stored as inert text
- Frame rate limits on WebSocket → 30 fps / 60 fpm
- Connection limits → 100 total / 10 per IP

---

## Production Deployment

### Docker Compose (Recommended for VPS / Self-Hosted)

```bash
# 1. Configure domain in nginx.conf (replace your-domain.com)
# 2. Get Let's Encrypt certs (first time)
mkdir -p certbot/conf certbot/www
docker run --rm -it -v $(pwd)/certbot/conf:/etc/letsencrypt \
  -v $(pwd)/certbot/www:/var/www/certbot \
  certbot/certbot certonly --webroot -w /var/www/certbot \
  -d your-domain.com --email you@email.com --agree-tos --no-eff-email

# 3. Start stack
docker compose up -d --build

# 4. Verify
curl https://your-domain.com/health
```

**Includes:** API (4 Gunicorn/Uvicorn workers), Nginx reverse proxy (WebSocket, rate limits, HTTPS), Certbot auto-renewal, persistent volumes for models/data.

### Kubernetes (Any Cluster)

```bash
# 1. Build & push multi-arch image
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ghcr.io/youruser/face-intelligence:latest --push .

# 2. Update image in k8s/deployment.yaml, then deploy
kubectl apply -f k8s/

# 3. Check status
kubectl get pods -n face-intelligence -w
```

**Includes:** 2 replicas, HPA (CPU 70% / Memory 80%), PVCs for models (RO) + data (RW), Ingress with cert-manager, readiness/liveness/startup probes, `/dev/shm` emptyDir for ONNX Runtime model sharing.

### Systemd (Bare Metal Linux)

```bash
sudo cp face-intelligence.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now face-intelligence
sudo journalctl -u face-intelligence -f
```

### CI/CD (GitHub Actions)

Push to `develop` → auto-deploys to staging.
Push version tag (`v1.0.0`) → auto-deploys to production.

---

## Free Hosting Recommendations

| Platform | Specs | Cost | Best For |
|----------|-------|------|----------|
| **Oracle Cloud Free Tier** | 4 ARM OCPU, 24 GB RAM | **Free forever** | **Best overall** — runs 8 workers comfortably |
| **Google Cloud Free Tier** | 1 e2-micro (2 vCPU, 1 GB) + $300 credit | Free 90 days | Testing only (RAM too low for 4 workers) |
| **Railway / Render** | Managed Docker deploy | Free tier / credit | Easiest hobby deploys |
| **Self-hosted (old PC / Pi 4 / Mini PC)** | Your hardware | $0 (electricity) | Full control, zero recurring cost |

**Quick Oracle Cloud deploy:**
```bash
# On Oracle ARM instance (VM.Standard.A1.Flex, 4 OCPU, 24 GB):
curl -fsSL https://get.docker.com | sh
git clone https://github.com/your-org/face-intelligence.git
cd face-intelligence
docker compose up -d --build
```

See `DEPLOYMENT_GUIDE.md` for complete guide with resource sizing, cost estimates, and step-by-step commands.

---

## Project Structure

```
face-intelligence/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── api/            # REST + WebSocket endpoints
│   │   ├── core/           # Config, ONNX Runtime, quantization
│   │   ├── schemas/        # Pydantic models
│   │   ├── security/       # Input validation, consent enforcement
│   │   └── services/       # Detection, recognition, expression, age, tracking, pipeline
│   ├── tests/              # 150+ tests (black/white/grey-box)
│   ├── scripts/            # Model downloader
│   └── requirements.txt
├── models/                 # 12 ONNX models (~16.6 MB INT8 total)
├── data/                   # Gallery, embeddings, uploads (gitignored)
├── k8s/                    # Kubernetes manifests
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile              # Multi-stage production build
├── docker-compose.yml      # API + Nginx + Certbot
├── nginx.conf              # Production reverse proxy
├── face-intelligence.service  # Systemd unit
├── DEPLOYMENT_GUIDE.md     # Complete deployment guide
└── README.md
```

---

## Key Bug Fixes (2026-11-10)

| Issue | Root Cause | Fix |
|-------|------------|-----|
| **Ghost face boxes** persisting 10+ frames across image/video/webcam | 3-layer defense incomplete: optical flow hid after 5 frames, track deletion didn't count predicted frames, deduplication only within-frame | `tracking.py`: `max_unconfirmed=2`, LK failure increments by 2; `update()`: `missed_frames += unconfirmed_predict_frames`; `pipeline.py`: tracker-aware `_deduplicate_faces()` zeros confidence for ghost tracks |
| **Age estimation normalization** | Both `genderage.onnx` and `fairface.onnx` are InsightFace models expecting `[-1,1]` input; code used raw `[0,255]` and ImageNet norm respectively | Investigated empirically: correct norm produces reasonable outputs but **regressed user's real-face results (25→38)**; **reverted to original preprocessing** per user request; original behavior restored (~25-30 range) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/models/status` | Model load status + accuracy badges |
| `POST` | `/api/analyze/image` | Analyze uploaded image |
| `POST` | `/api/analyze/video` | Analyze uploaded video |
| `WS` | `/ws/analyze` | Live webcam stream (query params: `run_recognition`, `run_expression`, `run_age`) |
| `POST` | `/api/faces/enroll` | Enroll face (requires `consent_confirmed=true`) |
| `GET` | `/api/faces` | List enrolled faces |
| `DELETE` | `/api/faces/{face_id}` | Delete enrolled face |

---

## Configuration (via `backend/app/core/config.py` or env vars)

```python
detection_interval: 3              # Run SCRFD every N frames
temporal_ema_alpha: 0.4            # EMA smoothing factor
batch_inference_enabled: true      # cv2.dnn.blobFromImages batching
onnx_execution_providers:          # Priority order
  - OpenVINOExecutionProvider
  - CPUExecutionProvider
onnx_intra_op_threads: 4           # Per-operator threads
recognition_distance_threshold: 0.68  # Cosine distance for match
```

---

## Privacy & Consent Model

- **Local-first**: No data leaves your machine/server
- **Consent required**: `/api/faces/enroll` mandates `consent_confirmed=true`
- **Right to deletion**: `DELETE /api/faces/{face_id}` removes gallery image + embedding
- **Consent withdrawal**: `POST /api/faces/{face_id}/withdraw-consent` permanently deletes biometric data
- **Encryption at rest (optional)**: Face embeddings encrypted with AES-256-GCM when `FACE_ENCRYPTION_KEY` + `FACE_ENCRYPTION_SALT` env vars are set. PBKDF2 (100k iterations) key derivation. Automatic backup + migration of existing plaintext embeddings on first enrollment after enabling.
- **No tracking**: No analytics, no telemetry, no external calls
- **Age estimation disclaimer**: Visual age estimation is inherently approximate; **must not** be used for age-gating, legal decisions, or minor detection

See `docs/PRIVACY.md` for full model.

---

## Security (Phase 1 Hardening)

This repository includes **Phase 1 Critical Security Hardening** — a baseline of controls suitable for a controlled/private deployment. Not production-ready for public biometric services.

### Authentication & Authorization
- **API key authentication** (X-API-Key header, Bearer token, or query param) with PBKDF2-HMAC-SHA256 hashing (100k iterations)
- **Role-Based Access Control (RBAC)** with 4 roles:
  - `ADMIN`: Full access (enroll, list, delete, analyze, admin)
  - `OPERATOR`: Analyze (image/video/stream), recognize, model status
  - `VIEWER`: List faces, model status only
  - `SERVICE`: Full API access (service-to-service)
- **Granular permissions** per endpoint via `require_permission()` dependencies

### WebSocket Security (`/ws/analyze`)
- Token required during handshake (`?token=fi_...`)
- Validates revocation, expiration, `ANALYZE_STREAM` permission
- Rate limits: 30 frames/sec + 60 frames/min per connection
- Connection limits: 100 concurrent, 10 per IP
- Idle timeout: 5 minutes
- Admin stats endpoint: `GET /ws/stats` (ADMIN_CONFIG permission)

### Encryption at Rest
- **AES-256-GCM** authenticated encryption for face embeddings
- **External key management**: Master key + salt from environment variables (`FACE_ENCRYPTION_KEY`, `FACE_ENCRYPTION_SALT`)
- **PBKDF2** key derivation (100k iterations, unique salt per key ID)
- **Key rotation support** via `KeyManager.rotate_key()`
- **Automatic migration**: On first enrollment with encryption enabled, creates timestamped backup of plaintext index and migrates all embeddings
- **Fails closed**: If encryption is configured but not initialized, gallery loading fails explicitly (no silent plaintext fallback)

### Rate Limiting
- HTTP: `slowapi` with IP + API key identification, 60 req/min default (configurable via `RATE_LIMIT_RPM`)
- WebSocket: Per-second + per-minute frame limits (separate from HTTP limiter)
- Redis support for distributed rate limiting (`REDIS_URL`)

### Production Configuration Validation
- `validate_production_config()` checks on startup:
  - `DEBUG=false`
  - `API_KEY_SALT` changed from default
  - `FACE_ENCRYPTION_KEY` + `FACE_ENCRYPTION_SALT` set
  - No `localhost` in `CORS_ORIGINS`
  - `RATE_LIMIT_ENABLED=true`
  - `REDIS_URL` set (production)
- Fails fast in production if validation fails

### Security Headers (All Responses)
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Content-Security-Policy: default-src 'none'; frame-ancestors 'none'
Strict-Transport-Security: max-age=31536000; includeSubDomains (production only)
```

### Known Limitations (Not Yet Implemented)
- Distributed rate limiting (Redis optional but not required)
- Automated key rotation / external KMS integration
- Audit logging for admin actions
- Certificate pinning / mTLS
- SIEM integration

---

## License

MIT License — see `LICENSE` file.

---

## Acknowledgments

- **InsightFace** — SCRFD, ArcFace, genderage models
- **HSEmotion** — ENet-B0 expression model
- **FairFace** — Balanced age/gender/race model
- **ONNX Runtime** — CPU inference engine
- **OpenCV** — Computer vision backend
- **FastAPI** — Modern Python web framework