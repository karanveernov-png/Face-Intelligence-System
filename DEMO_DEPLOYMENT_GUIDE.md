# Restricted Demo Deployment Guide
## Face Intelligence System - Security Hardened Configuration

**Purpose:** Deploy the Face Intelligence System as a temporary, evaluator-accessible HTTPS service with authentication and encryption at rest.

**Timeline:** Short-term demo (2-4 weeks) — not intended for long-term production use.

---

## Quick Start (5 minutes to first deploy)

### Prerequisites
- Docker and Docker Compose installed
- A registered domain name pointing to your server's IP
- SSH access to the target server

### Step 1: Generate Secret Keys

```bash
# Open a terminal and run these commands
cd /path/to/face-intelligence

# Generate API key salt (change from default immediately!)
API_SALT=$(python -c "import secrets; print(secrets.token_hex(32))")
echo "API_KEY_SALT=$API_SALT" >> .env

# Generate encryption keys for biometric data at rest
ENCRYPTION_KEY=$(python -c "import secrets,base64; print(base64.b64encode(secrets.token_bytes(32)).decode())")
ENCRYPTION_SALT=$(python -c "import secrets,base64; print(base64.b64encode(secrets.token_bytes(32)).decode())")
echo "FACE_ENCRYPTION_KEY=$ENCRYPTION_KEY" >> .env
echo "FACE_ENCRYPTION_SALT=$ENCRYPTION_SALT" >> .env

# Store the keys securely (they cannot be recovered if lost!)
cat > secure-keys.txt << EOF
Generated $(date):
API_KEY_SALT=$API_SALT
FACE_ENCRYPTION_KEY=$ENCRYPTION_KEY
FACE_ENCRYPTION_SALT=$ENCRYPTION_SALT
EOF
chmod 600 secure-keys.txt
```

### Step 2: Configure Domain

Edit `nginx.conf` and replace all occurrences of `YOUR_DOMAIN` with your actual domain:

```bash
# Replace YOUR_DOMAIN with your actual domain
sed -i 's/YOUR_DOMAIN/your-actual-domain.com/g' nginx.conf
```

Verify the replacement:
```bash
grep -n "YOUR_DOMAIN" nginx.conf
# Should return no results
grep -n "your-actual-domain.com" nginx.conf
# Should show line numbers where domain is configured
```

### Step 3: Provision TLS Certificate

```bash
# Create certbot directories
mkdir -p certbot/conf certbot/www

# Get Let's Encrypt certificate
docker compose run --rm certbot \
  certbot certonly --webroot \
  -w /var/www/certbot \
  -d your-actual-domain.com \
  --email admin@your-org.com \
  --agree-tos \
  --no-eff-email
```

Expected output on success:
```
IMPORTANT NOTES:
 - Congratulations! Your certificate and chain have been saved at...
 - Your cert will expire on...
 - To renew this certificate, use: certbot renew
```

### Step 4: Build and Start

```bash
# Build the Docker image with models included
docker compose build

# Start services
docker compose up -d

# Check health
curl https://your-actual-domain.com/health
# Expected: {"status":"ok","version":"0.3.0","environment":"production"}
```

### Step 5: Create Evaluator Credentials

```bash
# Generate API keys for evaluators
docker compose exec api python -c "
from app.security.auth import create_api_key, Role

evaluators = [
    ('evaluator1', Role.OPERATOR),
    ('evaluator2', Role.OPERATOR),
]

for name, role in evaluators:
    key, record = create_api_key(name=name, role=role, expires_in_days=7)
    print(f'{name}: {key}')
    print(f'  Hash: {record.key_hash[:16]}...')
"
```

Save the plaintext keys securely and distribute to evaluators via out-of-band channel (not email).

---

## Architecture Overview

```
                    ┌─────────────────────┐
                    │     INTERNET        │
                    └──────────┬──────────┘
                               │ HTTPS (443)
                    ┌──────────▼──────────┐
                    │      NGINX          │
                    │   Reverse Proxy     │
                    │   + Rate Limiting   │
                    │   + SSL Termination │
                    └──────────┬──────────┘
                               │ Internal (localhost)
                    ┌──────────▼──────────┐
                    │    FastAPI (Gunicorn)│
                    │    4 Workers × Uvicorn│
                    │    Port 8000 (local) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼────────┐ ┌────▼─────┐ ┌───────▼────────┐
    │  SCRFD-10G Model  │ │ArcFace-R50│ │ HSEmotion Model │
    │   Detection       │ │ Recognition│ │ Expression    │
    └──────────────────┘ └────────────┘ └───────────────┘
    
    Persistent Storage (Docker Volume):
    ┌──────────────────────────────────────┐
    │  /app/data/                          │
    │  ├── embeddings/index.json (encrypted)│
    │  ├── gallery/*.jpg (optional)        │
    │  └── tmp_uploads/ (cleaned after use) │
    └──────────────────────────────────────┘
```

---

## Security Controls in Place

### Authentication & Authorization
| Control | Implementation |
|---------|---------------|
| API Key Auth | X-API-Key header, Bearer token, or query param |
| PBKDF2 Hashing | 100,000 iterations SHA-256 |
| RBAC Roles | ADMIN, OPERATOR, VIEWER, SERVICE |
| Token Expiry | Optional expiration (set for demo: 7 days) |
| Revocation | In-memory revocation list (single-worker) |

### Data Protection
| Control | Implementation |
|---------|---------------|
| Encryption at Rest | AES-256-GCM for face embeddings |
| Consent Required | Enforced at enrollment endpoint |
| Right to Deletion | DELETE and withdraw-consent endpoints |
| No Logging of Biometrics | Raw images never written to logs |

### Network Security
| Control | Implementation |
|---------|---------------|
| TLS Only | HTTP → HTTPS redirect enforced |
| HSTS | max-age=31536000 includeSubDomains |
| CSP | default-src 'none'; frame-ancestors 'none' |
| CORS | Explicit origins only (no wildcards) |
| Rate Limiting | 30 req/s API, 10 req/s WebSocket, 5 req/min enroll |

### Container Security
| Control | Implementation |
|---------|---------------|
| Non-root User | UID 1000 (appuser) |
| Capability Drop | ALL capabilities dropped |
| Read-only Root | readOnlyRootFilesystem enabled in K8s |
| No Public API Port | Listens on 0.0.0.0:8000 inside container; docker-compose does NOT publish this port to the host — all external traffic routes through nginx over HTTPS |

---

## Evaluator Instructions

### Getting Access
1. Request an API key from the demo operator
2. Receive key via secure out-of-band channel (Signal, encrypted email, etc.)
3. Use key with any API endpoint via `X-API-Key` header

### Testing the API

```bash
# Health check (public)
curl https://your-domain.com/health

# Model status (public)
curl https://your-domain.com/api/models/status

# Analyze image (requires OPERATOR role)
curl -X POST "https://your-domain.com/api/analyze/image?run_age=true" \
  -H "X-API-Key: fi_your-key-here" \
  -F "file=@photo.jpg"

# List enrolled faces (requires LIST_FACES permission)
curl -H "X-API-Key: fi_your-key-here" \
  https://your-domain.com/api/faces

# Live stream (WebSocket, requires ANALYZE_STREAM permission)
wss://your-domain.com/ws/analyze?token=fi_your-key-here
```

### Frontend Access
Point browser to `https://your-domain.com`. Enter API key in the Settings panel or it can be set in the environment variables.

---

## Monitoring & Verification

### Health Checks
```bash
# All services healthy?
docker compose ps

# API responds?
curl -f https://your-domain.com/health && echo "OK"

# Models loaded correctly?
curl -s https://your-domain.com/api/models/status | jq '.[].loaded'
```

### Logs
```bash
# API logs
docker compose logs -f api

# Nginx access logs
docker compose logs -f nginx

# View recent errors
docker compose logs --tail=100 api | grep -i error
```

### Active Connections
```bash
# WebSocket connection count (admin endpoint)
curl -H "X-API-Key: fi_admin-key" https://your-domain.com/ws/stats
```

---

## Maintenance Procedures

### Regenerating Evaluation Keys
```bash
docker compose exec api python -c "
from app.security.auth import revoke_api_key, _hash_api_key, create_api_key, Role

# Revoke old key
old_key = 'fi_old-key-here'
revoke_api_key(_hash_api_key(old_key))
print('Old key revoked')

# Create new key
new_key, record = create_api_key(name='evaluator-1', role=Role.OPERATOR, expires_in_days=7)
print(f'New key: {new_key}')
"
```

### Scaling Considerations
This deployment uses a single Gunicorn worker process per container. For higher concurrency:

1. Increase workers in Dockerfile CMD: `"--workers", "8"`
2. Add Redis for distributed rate limiting: `REDIS_URL=redis://redis:6379`
3. Use Kubernetes deployment with multiple replicas (see k8s/deployment.yaml)

### Certificate Renewal
Certbot auto-renews every 12 hours. Verify:
```bash
docker compose exec certbot certbot certificates
```

---

## Decommissioning Procedure

When the demo period ends, securely remove all biometric data:

```bash
# 1. Stop all services
docker compose down

# 2. Remove persistent volumes (deletes all embeddings and galleries)
docker compose down -v

# 3. Alternatively, selectively delete only biometric data
docker volume rm face-intelligence_face-data
```

Data stored in the `face-data` volume is encrypted at rest and destroyed when the volume is removed.

---

## Known Limitations

1. **Single-worker API key store**: `create_api_key()` only persists in-process. Restart the container to clear keys. For multi-worker deployments, implement Redis-backed storage.

2. **Process-local rate limits**: SlowAPI tracks rates per-process. With multiple Gunicorn workers, effective limit is N× configured rate. Acceptable for limited-evaluator demos.

3. **No audit logging**: Administrative actions (enroll, delete, view) are not currently logged to a separate audit trail. Plan for addition before production expansion.

4. **Frontend esbuild vulnerability**: npm audit shows moderate severity in devDependency. Does not affect production bundle but consider updating vite dependency when convenient.

---

## Success Criteria

- [ ] HTTPS accessible at configured domain
- [ ] Health endpoint returns 200 OK
- [ ] API key authentication rejects invalid keys (401)
- [ ] WebSocket requires valid token (rejects missing/invalid)
- [ ] Enrollment requires consent confirmation
- [ ] Encryption at rest active (check logs: "Encryption initialized successfully")
- [ ] All evaluations complete within demo timeframe
- [ ] Post-demo cleanup completed (volumes removed)

---

**Document Version:** 1.0  
**Last Updated:** November 11, 2026  
**Repository:** https://github.com/karanveernov-png/Face-Intelligence-System
