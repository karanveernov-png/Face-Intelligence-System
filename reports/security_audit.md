# Security Audit Report

**Project:** AI Face Intelligence System  
**Audit Date:** 2026-11-10  
**Classification:** CONFIDENTIAL — Internal Use Only  

---

## Scope

This audit covers the Face Intelligence System backend (FastAPI + ONNX Runtime), including:
- REST API endpoints (`/api/analyze/*`, `/api/faces/*`)
- WebSocket endpoint (`/ws/analyze`)
- Model loading and inference pipeline
- File upload handling and temporary storage
- Biometric data storage (face embeddings gallery)
- Deployment infrastructure (Docker, Kubernetes, CI/CD)

**Out of Scope:** Frontend application, physical security of deployment hardware, third-party model provenance beyond checksum verification.

---

## Methodology

1. **Static Code Analysis** — Manual review of all Python source files
2. **Dependency Review** — Requirements.txt audit, known vulnerability check
3. **Configuration Review** — Docker, docker-compose, Kubernetes manifests
4. **Test Suite Analysis** — Existing security tests in `tests/test_security_blackbox.py`
5. **Threat Modeling** — STRIDE analysis on data flows

---

## Findings Summary

| Severity | Count | Description |
|----------|-------|-------------|
| **P0 — Critical** | 4 | Immediate risk of data exposure or system compromise |
| **P1 — High** | 5 | Significant security gaps requiring prompt remediation |
| **P2 — Medium** | 4 | Defense-in-depth improvements |
| **P3 — Low/Info** | 4 | Documentation and hardening |

---

## Detailed Findings

### P0 — Critical

#### SEC-001: No Authentication or Authorization on Any Endpoint
**CVSS 3.1: 9.8 (Critical)** — AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H

**Affected Files:** `app/main.py`, `app/api/routes.py`, `app/api/ws.py`

**Description:** All API endpoints are publicly accessible without any authentication mechanism. Any network caller can:
- Enroll faces into the gallery (`POST /api/faces/enroll`)
- List all enrolled identities (`GET /api/faces`)
- Delete any face by ID (`DELETE /api/faces/{face_id}`)
- Analyze images/videos (`POST /api/analyze/image`, `POST /api/analyze/video`)
- Stream live webcam frames (`WS /ws/analyze`)

**Impact:** Complete compromise of biometric database. Attackers can enumerate, add, or delete identities. WebSocket allows unauthorized real-time face analysis.

**Reproduction:**
```bash
# Enroll a face without auth
curl -X POST http://localhost:8000/api/faces/enroll \
  -F "file=@attacker.jpg" -F "name=Victim" -F "consent_confirmed=true"

# List all enrolled faces
curl http://localhost:8000/api/faces

# Delete a face
curl -X DELETE http://localhost:8000/api/faces/{face_id}
```

**Remediation:**
1. Implement API key authentication middleware (`app/security/auth.py`)
2. Require valid API key on all `/api/*` routes
3. Require token in WebSocket connection (query param or subprotocol)
4. Add role-based access control (admin vs read-only)

**Verification:** Integration test confirming 401/403 without valid credentials.

---

#### SEC-002: Biometric Embeddings Stored in Plaintext
**CVSS 3.1: 8.1 (High)** — AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:N

**Affected Files:** `app/services/recognition.py` (`_save_index`, `_load_index`), `data/embeddings/index.json`

**Description:** Face embeddings (512-dimensional vectors uniquely identifying individuals) are stored in a plaintext JSON file at `data/embeddings/index.json` with no encryption at rest.

**Impact:** 
- File system access = full biometric database compromise
- Embeddings are irreversible — unlike passwords, they cannot be rotated
- GDPR/CCPA/biometric privacy law violations (BIPA, etc.)

**Evidence:**
```json
// data/embeddings/index.json
{
  "uuid-here": {
    "name": "John Doe",
    "embedding": [0.123, -0.456, ...],  // 512 floats
    "model": "ArcFace-R50-INT8",
    "enrolled_at": 1731234567.89
  }
}
```

**Remediation:**
1. Encrypt embeddings at rest using AES-256-GCM with per-deployment key
2. Store encryption key in environment variable / secret manager (not in repo)
3. Implement key rotation strategy
4. Consider hardware security module (HSM) for production

**Verification:** Confirm `index.json` contains encrypted blobs, not raw vectors.

---

#### SEC-003: WebSocket Endpoint Accepts Unauthenticated Connections
**CVSS 3.1: 9.1 (Critical)** — AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:H

**Affected Files:** `app/api/ws.py` (`analyze_stream`)

**Description:** The WebSocket endpoint `/ws/analyze` accepts connections without any authentication. Clients can stream arbitrary frames for real-time face analysis.

**Impact:**
- Unauthorized use of compute resources (GPU/CPU inference)
- DoS via frame flooding (no rate limiting per connection)
- Privacy violation — anyone can send frames for analysis
- Potential for model extraction attacks

**Reproduction:**
```python
import websockets
async with websockets.connect("ws://localhost:8000/ws/analyze") as ws:
    await ws.send(frame_bytes)  # No auth required
```

**Remediation:**
1. Require authentication token in WebSocket handshake:
   - Query parameter: `ws://host/ws/analyze?token=API_KEY`
   - Or subprotocol negotiation
2. Validate token before `websocket.accept()`
3. Apply per-connection rate limiting

**Verification:** Connection rejected without valid token.

---

#### SEC-004: CORS Configuration Hardcoded to Localhost
**CVSS 3.1: 5.3 (Medium)** — AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N

**Affected Files:** `app/main.py` (lines 25-30)

**Description:** CORS middleware allows only `http://localhost:5173` and `http://127.0.0.1:5173`. Production deployments must modify source code to add their domain.

**Impact:** 
- Forces code changes for deployment (anti-pattern)
- Risk of developers setting `allow_origins=["*"]` to "fix" it
- No validation that origin matches expected deployment domain

**Remediation:**
1. Make CORS origins configurable via environment variable
2. Parse comma-separated list at startup
3. Validate configuration on startup (fail fast if misconfigured)
4. Document production CORS setup

**Verification:** Deploy with custom origin via env var, verify CORS headers.

---

### P1 — High

#### SEC-005: No Rate Limiting on Any Endpoint
**CVSS 3.1: 7.5 (High)** — AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H

**Affected Files:** `app/main.py`, `app/api/routes.py`, `app/api/ws.py`

**Description:** No rate limiting exists on:
- Image upload (`POST /api/analyze/image`)
- Video upload (`POST /api/analyze/video`)
- Face enrollment (`POST /api/faces/enroll`)
- WebSocket frame processing

**Impact:** 
- DoS via resource exhaustion (CPU inference, memory, disk)
- Brute-force face enrollment
- WebSocket frame flood crashes server

**Remediation:**
1. Add `slowapi` or custom rate limiter middleware
2. Configure per-endpoint limits:
   - Image: 30 req/min per IP
   - Video: 5 req/min per IP
   - Enrollment: 10 req/min per IP
   - WebSocket: 10 frames/sec per connection
3. Use Redis-backed limiter for multi-worker deployments

**Verification:** Load test confirms 429 responses under limit.

---

#### SEC-006: Missing Structured Logging and Request Tracing
**CVSS 3.1: 5.3 (Medium)** — AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N

**Affected Files:** `app/main.py` (line 11), all service modules

**Description:** Uses `logging.basicConfig(level=logging.INFO)` only. No:
- Request IDs for traceability
- Structured JSON logs
- Audit logging for sensitive operations (enroll, delete, recognize)
- Correlation across async boundaries (WebSocket frames)

**Impact:** 
- Cannot investigate incidents or audit access
- No compliance evidence for biometric data access
- Debugging production issues is extremely difficult

**Remediation:**
1. Integrate `structlog` with JSON output
2. Add middleware generating request IDs (UUID)
3. Log all sensitive operations with: user, action, resource, timestamp, result
4. Ensure WebSocket frames carry correlation ID

**Verification:** Structured logs appear in container output with request IDs.

---

#### SEC-007: Consent Only Enforced at Enrollment Time
**CVSS 3.1: 6.5 (Medium)** — AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N

**Affected Files:** `app/api/routes.py` (enroll, list, delete), `app/services/recognition.py`

**Description:** The `consent_confirmed=true` check exists only in `POST /api/faces/enroll`. Subsequent operations (`GET /api/faces`, `DELETE /api/faces/{id}`, recognition matching) do not re-verify consent.

**Impact:** 
- Consent can be "bypassed" by enrolling with consent, then using face without ongoing consent
- No audit trail of consent status per operation
- Violates "ongoing consent" principle in biometric regulations

**Remediation:**
1. Store consent timestamp and version with each enrollment
2. Add consent verification middleware for recognition endpoints
3. Implement consent withdrawal endpoint that purges data
4. Audit log all recognition matches with consent status

**Verification:** Recognition request fails if enrollment consent withdrawn.

---

#### SEC-008: No Model File Integrity Verification
**CVSS 3.1: 7.4 (High)** — AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H

**Affected Files:** `app/core/onnx_runtime.py` (`resolve_model_path`, `load_onnx_session`)

**Description:** Model files (`.onnx`) are loaded from disk without any checksum verification. An attacker with filesystem access can replace models with malicious versions.

**Impact:** 
- Supply chain attack: swapped model = arbitrary code execution (ONNX supports custom ops)
- Model poisoning: degraded accuracy, backdoors
- No detection of corrupted downloads

**Remediation:**
1. Generate SHA256 checksums for all models at build time
2. Store checksums in signed manifest (`models/manifest.json.sig`)
3. Verify checksums in `load_onnx_session()` before loading
4. Fail startup if any model hash mismatches

**Verification:** Startup fails with tampered model file.

---

#### SEC-009: Video Processing Parameters Not Validated
**CVSS 3.1: 5.3 (Medium)** — AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L

**Affected Files:** `app/api/routes.py` (`analyze_video`), `app/core/config.py`

**Description:** Video processing uses `settings.frame_skip` and `settings.max_video_frames_processed` without validation. A malicious upload could exploit edge cases.

**Impact:** Resource exhaustion via crafted video parameters.

**Remediation:** Add validation bounds in config (e.g., `frame_skip >= 1`, `max_video_frames_processed <= 1000`).

---

### P2 — Medium

#### SEC-010: Generic Error Handling May Leak Stack Traces
**Affected Files:** `app/main.py`, all route handlers

**Description:** No global exception handler to sanitize errors in production. FastAPI's default debug handler may leak stack traces.

**Remediation:** Add global exception handler that returns generic errors in production mode.

#### SEC-011: Missing Security Headers (Application Layer)
**Affected Files:** `app/main.py`, `nginx.conf`

**Description:** While nginx provides some headers, the FastAPI app should also set:
- `Content-Security-Policy`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`

**Remediation:** Add `Starlette` middleware for security headers.

#### SEC-012: Temporary Upload Directory Permissions
**Affected Files:** `app/core/config.py` (line 11), `app/security/validation.py`

**Description:** `TMP_UPLOAD_DIR` created with default permissions (world-readable on some systems). Uploaded files may be readable by other users on shared systems.

**Remediation:** Set directory mode to `0o700` on creation. Clean up aggressively.

#### SEC-013: No Automated Dependency Vulnerability Scanning
**Affected Files:** `.github/workflows/ci-cd.yml`

**Description:** CI/CD pipeline lacks:
- `pip-audit` or `safety` scan
- `Dependabot` alerts
- SBOM generation (`cyclonedx-bom`)
- Container image scanning (`trivy`)

**Remediation:** Add dependency scanning job to CI pipeline.

---

### P3 — Low / Informational

#### SEC-014: No API Versioning Strategy
**Impact:** Breaking changes require coordinated client/server updates.

#### SEC-015: Debug Endpoints Not Explicitly Disabled
**Impact:** `/docs`, `/redoc`, `/openapi.json` exposed in production.

#### SEC-016: No Security Headers in WebSocket Handshake
**Impact:** Limited browser-side protection for WS connections.

#### SEC-017: Consent Text Not Versioned
**Impact:** Cannot prove which consent version user agreed to.

---

## Threat Model (STRIDE)

| Threat | Vectors | Mitigations (Current) | Gaps |
|--------|---------|----------------------|------|
| **Spoofing** | No auth on any endpoint | None | **SEC-001, SEC-003** |
| **Tampering** | Model file replacement, embedding modification | None | **SEC-002, SEC-008** |
| **Repudiation** | No audit logging | None | **SEC-006** |
| **Information Disclosure** | Plaintext embeddings, stack traces | Validation prevents decompression bombs | **SEC-002, SEC-010** |
| **Denial of Service** | No rate limits, unbounded video processing | Max upload size, decoded pixel limits | **SEC-005, SEC-009** |
| **Elevation of Privilege** | N/A (no auth = no privilege separation) | N/A | **SEC-001** |

---

## Compliance Assessment

| Regulation | Requirement | Status | Gap |
|------------|-------------|--------|-----|
| **GDPR Art. 32** | Encryption at rest | ❌ FAIL | SEC-002 |
| **GDPR Art. 25** | Data protection by design | ⚠️ PARTIAL | SEC-001, SEC-007 |
| **BIPA (Illinois)** | Written consent, retention policy | ⚠️ PARTIAL | SEC-007, no auto-delete |
| **CCPA** | Right to deletion | ✅ PASS | DELETE endpoint exists |
| **SOC 2** | Access control, audit logging | ❌ FAIL | SEC-001, SEC-006 |

---

## Remediation Priority & Effort

| ID | Effort | Priority | Dependencies |
|----|--------|----------|--------------|
| SEC-001 | Medium (2-3 days) | P0 | None |
| SEC-002 | Medium (2-3 days) | P0 | Key management strategy |
| SEC-003 | Low (1 day) | P0 | SEC-001 |
| SEC-004 | Low (0.5 day) | P0 | None |
| SEC-005 | Medium (2 days) | P1 | Redis for distributed |
| SEC-006 | Medium (2 days) | P1 | Structlog integration |
| SEC-007 | Medium (2 days) | P1 | SEC-001 |
| SEC-008 | Low (1 day) | P1 | Build-time manifest generation |
| SEC-009 | Low (0.5 day) | P1 | None |
| SEC-010 | Low (0.5 day) | P2 | None |
| SEC-011 | Low (0.5 day) | P2 | None |
| SEC-012 | Low (0.5 day) | P2 | None |
| SEC-013 | Medium (1 day) | P2 | CI pipeline modification |

---

## Verification Checklist for Remediation

- [ ] All P0 findings have fix implemented and tested
- [ ] Integration tests confirm 401/403 without auth
- [ ] Embeddings file contains encrypted data only
- [ ] WebSocket rejects connections without token
- [ ] Rate limiter returns 429 under load
- [ ] Structured JSON logs with request IDs in container output
- [ ] Model checksum verification fails on tampered file
- [ ] Dependency scan passes in CI
- [ ] Security headers present in HTTP responses
- [ ] Penetration test (internal) completed

---

## Conclusion

The Face Intelligence System has **strong computer vision engineering** but **critical security architecture gaps**. The absence of authentication, encryption at rest, and rate limiting makes it unsuitable for any network-exposed deployment.

**Immediate Action Required:** Implement P0 fixes (SEC-001 through SEC-004) before any deployment beyond localhost.

**Estimated Remediation Timeline:** 2-3 weeks for P0+P1, 4-6 weeks for full hardening including penetration testing.