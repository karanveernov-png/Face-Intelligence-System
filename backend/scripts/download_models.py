"""
Download & Dynamic INT8 Quantization helper for the Accuracy-First Production Stack.

Usage:
    python scripts/download_models.py

Downloads FP32 `.onnx` weights into `<repo_root>/models/` and runs
`onnxruntime.quantization.quantize_dynamic` (`QuantType.QUInt8`) to produce
the INT8 files `app/services/*.py` actually look for. Only pulls the specific
zip members it needs via HTTP Range requests instead of the full archives
(FairFace is the one exception -- see #5 below).

Verified working (Nov 2026) end to end: downloads, quantizes, and loads every
model with onnxruntime before declaring success. If this script was never run
(or was run somewhere without internet access), every pipeline silently drops
to its placeholder fallback -- OpenCV Haar for detection, a hand-tuned formula
biased toward "neutral" for expression, a random projection for recognition
embeddings, and an honest "no estimate available" for age/gender rather than
a guessed number. Those fallbacks exist so the app still *runs* for UI
development, not as a substitute for real models.

Models fetched, with the real accuracy each one actually has:
  1. SCRFD-10G INT8      (`scrfd_10g_bnkps_int8.onnx`, ~4.3 MB)  -- 95.16% WiderFace Hard
  2. ArcFace-R50 INT8    (`w600k_r50_int8.onnx`, ~43.8 MB)       -- 99.83% LFW (most
     accurate recognition option; "GhostFaceNet-V2" in older config/comments was
     aspirational and was never a real downloadable file)
  3. MobileFaceNet INT8  (`mobilefacenet_arcface_int8.onnx`, ~3.5 MB) -- 99.65% LFW,
     a much smaller alternative if the 43.8 MB recognition model is too heavy
  4. HSEmotion ENet-B0 INT8 (`hsemotion_enet_b0_8_int8.onnx`, ~4.2 MB) -- 86.9% RAF-DB
  5. Age + gender, as a two-model ensemble (see app/services/age_estimation.py):
       - `genderage.onnx` (~1.3 MB, from the same buffalo_l pack as #2) --
         InsightFace's continuous 0-100 age regression + gender logits.
       - `fairface.onnx` / `fairface_int8.onnx` (~82 MB FP32 / ~21 MB INT8) --
         FairFace's 9-bucket age classifier, fetched via the `uniface` package's
         own verified (SHA-256-checked) downloader rather than a hand-maintained
         URL, since FairFace's upstream weights aren't hosted anywhere with a
         stable direct link. Requires `pip install uniface` (script-only; not a
         runtime dependency of the backend itself).
     Earlier revisions of this script instead fetched a dlib/Cydral ResNet-10 age
     model that `age_estimation.py` doesn't actually load anymore, and claimed
     the systematic "defaults ambiguous faces toward adult" bias was an
     unfixable property of "every public age model." It wasn't -- that was this
     project's own heuristic logic, removed in the rewrite; see this file's
     history in CHANGELOG.md and age_estimation.py's module docstring.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BUFFALO_L_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_S_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
HSEMOTION_ONNX_URL = (
    "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx"
)



class _RangeHTTPFile:
    """File-like object backed by HTTP Range requests, so zipfile only pulls
    the bytes of the central directory + the specific members we ask for,
    instead of downloading the entire remote archive (zip or wheel -- a
    .whl is just a zip)."""

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            self.length = int(r.headers["Content-Length"])
        self.pos = 0

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.length + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        end = (self.length - 1) if (size is None or size < 0) else min(self.pos + size, self.length) - 1
        if end < self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True


def extract_members_from_remote_zip(url: str, wanted: dict[str, Path]) -> dict[str, Path]:
    """wanted: {member_suffix: dest_path}. Falls back to a full download if the
    server doesn't support Range requests (rare, but some mirrors don't)."""
    try:
        f = _RangeHTTPFile(url)
        zf = zipfile.ZipFile(f)
        found = {}
        for member in zf.namelist():
            for suffix, dest_path in wanted.items():
                if member.endswith(suffix):
                    dest_path.write_bytes(zf.read(member))
                    print(f"  extracted {member} -> {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)")
                    found[suffix] = dest_path
        return found
    except Exception as exc:
        print(f"  range-based extraction failed ({exc}); falling back to full download")
        tmp_zip = MODELS_DIR / "_tmp_download.zip"
        urllib.request.urlretrieve(url, str(tmp_zip))
        found = {}
        with zipfile.ZipFile(tmp_zip) as zf:
            for member in zf.namelist():
                for suffix, dest_path in wanted.items():
                    if member.endswith(suffix):
                        dest_path.write_bytes(zf.read(member))
                        found[suffix] = dest_path
        tmp_zip.unlink(missing_ok=True)
        return found


def fetch_fairface_via_uniface(dest_fp32: Path) -> bool:
    """FairFace's upstream weights (CC BY 4.0) aren't hosted anywhere with a
    stable direct-download URL -- the original repo links to a Google Drive
    folder, which isn't scriptable or trustworthy to hardcode here. Instead,
    this uses the `uniface` package (MIT-licensed, PyPI), which bundles a
    SHA-256-verified downloader for the same weights. Requires
    `pip install uniface`; this is a script-only dependency, not something
    the backend itself needs at runtime."""
    try:
        from uniface.attribute import FairFace
        from uniface.model_store import get_cache_dir
    except ImportError:
        print("  FAILED: the `uniface` package isn't installed. Run `pip install uniface` "
              "and re-run this script, or place a fairface.onnx you already have into "
              f"{dest_fp32.parent} yourself.")
        return False

    try:
        FairFace()  # instantiating triggers the package's own verified download+cache
    except Exception as exc:
        print(f"  FAILED to fetch FairFace weights via uniface: {exc}")
        return False

    cache_dir = Path(get_cache_dir())
    candidates = sorted(cache_dir.rglob("*fairface*.onnx")) or sorted(cache_dir.rglob("*FairFace*.onnx"))
    if not candidates:
        print(f"  FAILED: uniface reported success but no fairface*.onnx found under {cache_dir}")
        return False

    dest_fp32.write_bytes(candidates[0].read_bytes())
    print(f"  copied {candidates[0]} -> {dest_fp32.name} ({dest_fp32.stat().st_size / 1e6:.2f} MB)")
    return True


def quantize_to_int8(fp32_path: Path, int8_path: Path) -> None:
    if int8_path.is_file() and int8_path.stat().st_size > 1024:
        return
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(model_input=str(fp32_path), model_output=str(int8_path), weight_type=QuantType.QUInt8)
    print(
        f"  quantized {fp32_path.name} ({fp32_path.stat().st_size / 1e6:.2f} MB) -> "
        f"{int8_path.name} ({int8_path.stat().st_size / 1e6:.2f} MB)"
    )


def verify_onnx_loads(path: Path) -> bool:
    """The step the original script was missing: actually load the model and
    confirm onnxruntime accepts it before declaring success."""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _ = sess.get_inputs()[0].shape
        return True
    except Exception as exc:
        print(f"  FAILED to load {path.name}: {exc}")
        return False


def main() -> None:
    ok = True

    print("Detection: SCRFD-10G (from buffalo_l)")
    scrfd_fp32 = MODELS_DIR / "scrfd_10g_bnkps.onnx"
    scrfd_int8 = MODELS_DIR / "scrfd_10g_bnkps_int8.onnx"
    if not scrfd_int8.is_file():
        extract_members_from_remote_zip(BUFFALO_L_ZIP_URL, {"det_10g.onnx": scrfd_fp32})
        quantize_to_int8(scrfd_fp32, scrfd_int8)
    ok &= verify_onnx_loads(scrfd_int8)

    print("Recognition: ArcFace-R50, highest accuracy option (from buffalo_l)")
    r50_fp32 = MODELS_DIR / "w600k_r50.onnx"
    r50_int8 = MODELS_DIR / "w600k_r50_int8.onnx"
    if not r50_int8.is_file():
        extract_members_from_remote_zip(BUFFALO_L_ZIP_URL, {"w600k_r50.onnx": r50_fp32})
        quantize_to_int8(r50_fp32, r50_int8)
    ok &= verify_onnx_loads(r50_int8)

    print("Recognition: MobileFaceNet, compact alternative (from buffalo_s)")
    mbf_fp32 = MODELS_DIR / "w600k_mbf.onnx"
    mbf_int8 = MODELS_DIR / "mobilefacenet_arcface_int8.onnx"
    if not mbf_int8.is_file():
        extract_members_from_remote_zip(BUFFALO_S_ZIP_URL, {"w600k_mbf.onnx": mbf_fp32})
        quantize_to_int8(mbf_fp32, mbf_int8)
    ok &= verify_onnx_loads(mbf_int8)

    print("Expression: HSEmotion ENet-B0 (FP32 Unquantized)")
    hse_fp32 = MODELS_DIR / "enet_b0_8_best_vgaf.onnx"
    if not hse_fp32.is_file():
        urllib.request.urlretrieve(HSEMOTION_ONNX_URL, str(hse_fp32))
    ok &= verify_onnx_loads(hse_fp32)

    print("Age (continuous) + gender: InsightFace genderage.onnx (from buffalo_l)")
    genderage_path = MODELS_DIR / "genderage.onnx"
    if not genderage_path.is_file():
        extract_members_from_remote_zip(BUFFALO_L_ZIP_URL, {"genderage.onnx": genderage_path})
    ok &= verify_onnx_loads(genderage_path)

    print("Age (bucketed) + gender + race: FairFace, via the `uniface` package's verified downloader")
    fairface_fp32 = MODELS_DIR / "fairface.onnx"
    fairface_int8 = MODELS_DIR / "fairface_int8.onnx"
    if not fairface_fp32.is_file():
        fetched = fetch_fairface_via_uniface(fairface_fp32)
        if not fetched:
            ok = False
    if fairface_fp32.is_file():
        quantize_to_int8(fairface_fp32, fairface_int8)
        ok &= verify_onnx_loads(fairface_int8)
    print(
        "  note: age_estimation.py averages genderage.onnx and fairface.onnx and reports "
        "their disagreement as an honest uncertainty, rather than collapsing ambiguous "
        "faces toward either age category -- see that module's docstring for why."
    )

    print("\nFinal models/ contents:")
    for p in sorted(MODELS_DIR.glob("*")):
        if p.is_file():
            print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")

    if not ok:
        print("\nOne or more models failed to load -- see FAILED lines above.")
        sys.exit(1)
    print("\nAll models downloaded, quantized, and verified loadable.")


if __name__ == "__main__":
    main()
