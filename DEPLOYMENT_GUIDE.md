# Free Deployment Options for Face Intelligence

## Overview
Your stack: **FastAPI + ONNX Runtime (CPU) + OpenCV** | ~16 MB models | 2-4 GB RAM | 2-4 vCPU

---

## Tier 1: Always Free (No Time Limit)

| Platform | Specs | Best For | Limitations |
|----------|-------|----------|-------------|
| **Oracle Cloud Free Tier** | **4 ARM Ampere cores, 24 GB RAM** (2x VM.Standard.A1.Flex) | **Best overall** - Native ARM64, plenty of RAM for models + workers | Requires credit card; ARM64 (your Dockerfile supports `linux/arm64`) |
| **Google Cloud Free Tier** | 1 e2-micro (2 vCPU, 1 GB RAM) + $300 credit | Testing only | 1 GB RAM too low for 4 workers; use 1 worker |
| **AWS Free Tier** | 1 t3.micro (2 vCPU, 1 GB RAM) for 12 months | Testing only | 1 GB RAM too low; 12-month limit |
| **Azure Free Tier** | 1 B1S (1 vCPU, 1 GB RAM) for 12 months | Testing only | 1 GB RAM too low; 12-month limit |
| **Fly.io** | 3 shared-cpu-1x VMs, 256 MB each + 3 GB persistent volume | Small apps | 256 MB too low for ONNX Runtime; need paid tier |
| **Railway** | $5/mo credit (effectively free for hobby) | Easy deploys | Credit expires; then pay-as-you-go |
| **Render** | Free tier: 512 MB RAM, spins down after inactivity | Simple web services | Spins down (cold start ~30s); 512 MB tight |

---

## Tier 2: Generous Free Credits (Time-Limited)

| Platform | Credit | Duration | Notes |
|----------|--------|----------|-------|
| **Google Cloud** | $300 | 90 days | Can run 2-4 e2-standard-4 (4 vCPU, 16 GB) |
| **AWS** | $100+ | 12 months | Limited instance types |
| **Azure** | $200 | 30 days | Good GPU options if needed later |
| **DigitalOcean** | $200 | 60 days | Via GitHub Student Pack or referrals |
| **Vultr** | $100 | 30 days | Good bare metal options |
| **Linode/Akamai** | $100 | 60 days | Simple pricing |

---

## Tier 3: Self-Hosted (Zero Cost If You Have Hardware)

| Option | Hardware Needed | Pros |
|--------|----------------|------|
| **Old laptop/desktop** | Any x86_64 with 8+ GB RAM | Zero cost, full control |
| **Raspberry Pi 4/5 (8 GB)** | ~$100 | Low power, ARM64 native |
| **Mini PC (Beelink, Minisforum)** | $150-300 | x86_64, 16-32 GB RAM, 4-8 cores |
| **Intel NUC** | $200-400 | Compact, powerful |
| **Home lab server** | Any | Best performance |

**Recommended self-hosted setup:**
```bash
# On your machine (Linux):
git clone your-repo
cd face-intelligence
docker compose up -d
# Access at http://your-lan-ip:8000
```

---

## My Recommendations

### **Best Free Production: Oracle Cloud Free Tier**
```bash
# 1. Sign up at cloud.oracle.com (needs credit card, no charge)
# 2. Create 2x VM.Standard.A1.Flex (4 OCPU, 24 GB total)
# 3. Install Docker: curl -fsSL https://get.docker.com | sh
# 4. Deploy:
git clone your-repo
cd face-intelligence
docker compose up -d
# 5. Configure nginx + Let's Encrypt (certbot)
```
**Why:** 4 ARM64 cores + 24 GB RAM = runs 8 workers comfortably. Always free.

### **Easiest Hobby: Railway / Render**
- Connect GitHub repo → auto-deploys on push
- Railway: `docker build` from Dockerfile
- Render: Uses `docker-compose.yml` or Dockerfile
- **Caveat:** Not truly free forever; good for demos

### **Best for Learning K8s: Kind / k3d / Minikube (Local)**
```bash
# Local Kubernetes cluster
kind create cluster
kubectl apply -f k8s/
# Or k3d (lighter):
k3d cluster create face-intel --port "80:80@loadbalancer" --port "443:443@loadbalancer"
```

---

## Quick Start Commands

### Docker Compose (VPS / Self-hosted)
```bash
# 1. Clone & configure
git clone https://github.com/your/repo.git
cd face-intelligence
cp .env.example .env  # Edit domain, secrets

# 2. Get HTTPS certs (first time only)
mkdir -p certbot/conf certbot/www
docker run --rm -it -v $(pwd)/certbot/conf:/etc/letsencrypt \
  -v $(pwd)/certbot/www:/var/www/certbot \
  certbot/certbot certonly --webroot -w /var/www/certbot \
  -d your-domain.com --email you@email.com --agree-tos --no-eff-email

# 3. Start
docker compose up -d --build

# 4. Verify
curl https://your-domain.com/health
```

### Kubernetes (Oracle Cloud / Any K8s)
```bash
# 1. Build & push image
docker build -t ghcr.io/youruser/face-intelligence:latest .
docker push ghcr.io/youruser/face-intelligence:latest

# 2. Update k8s/deployment.yaml with your image
sed -i 's|your-registry/face-intelligence:latest|ghcr.io/youruser/face-intelligence:latest|g' k8s/deployment.yaml

# 3. Deploy
kubectl apply -f k8s/

# 4. Check status
kubectl get pods -n face-intelligence -w
kubectl logs -n face-intelligence -l app=face-intelligence-api -f
```

### Systemd (Bare Metal Linux)
```bash
# 1. Create user & directories
sudo useradd -r -s /bin/bash -d /opt/face-intelligence faceintel
sudo mkdir -p /opt/face-intelligence/{data,models}
sudo chown -R faceintel:faceintel /opt/face-intelligence

# 2. Deploy code
sudo -u faceintel git clone https://github.com/your/repo.git /opt/face-intelligence/app
sudo -u faceintel python3 -m venv /opt/face-intelligence/.venv
sudo -u faceintel /opt/face-intelligence/.venv/bin/pip install -r /opt/face-intelligence/app/backend/requirements.txt

# 3. Copy models
sudo cp -r models/* /opt/face-intelligence/models/

# 4. Install service
sudo cp face-intelligence.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now face-intelligence

# 5. Check
sudo journalctl -u face-intelligence -f
```

---

## Resource Requirements Summary

| Deployment | CPU | RAM | Storage | Models |
|------------|-----|-----|---------|--------|
| **Minimal (1 worker)** | 1 vCPU | 1.5 GB | 2 GB | Loaded per worker |
| **Recommended (4 workers)** | 2-4 vCPU | 3-4 GB | 5 GB | Shared via `/dev/shm` |
| **High Load (8 workers)** | 4-8 vCPU | 6-8 GB | 10 GB | Shared via `/dev/shm` |

**Key:** Always allocate `/dev/shm` (tmpfs) ≥ 512 MB for ONNX Runtime memory-mapped model sharing across workers.

---

## Cost Estimate (Paid Options)

| Provider | Instance | Monthly | Notes |
|----------|----------|---------|-------|
| **Hetzner CX42** | 8 vCPU, 16 GB, 160 GB SSD | **€32** | Best price/performance |
| **DigitalOcean 4GB** | 2 vCPU, 4 GB, 80 GB SSD | $24 | Simple, predictable |
| **Linode 4GB** | 2 vCPU, 4 GB, 80 GB SSD | $24 | Good support |
| **Vultr High Freq 4GB** | 2 vCPU, 4 GB, 100 GB NVMe | $24 | Fast NVMe |
| **Oracle Cloud** | 4 OCPU, 24 GB | **Free** | ARM64 only |

---

## Next Steps

1. **Choose platform** based on your needs above
2. **Fork this repo** and add your domain to `nginx.conf`
3. **Set up CI/CD** (GitHub Actions included) for auto-deploy
4. **Monitor** with `curl https://your-domain.com/health` and `kubectl logs` / `docker compose logs`

The deployment files are ready in your repo:
- `Dockerfile` - Multi-stage optimized build
- `docker-compose.yml` - Local/VPS with nginx + certbot
- `nginx.conf` - Production-ready reverse proxy with WebSocket, rate limits, HTTPS
- `k8s/deployment.yaml` - Full K8s with HPA, PVCs, Ingress
- `.github/workflows/ci-cd.yml` - Automated build/test/deploy
- `face-intelligence.service` - Systemd alternative