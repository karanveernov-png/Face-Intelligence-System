import { useRef, useState } from "react";
import { enrollFace, ApiError } from "../services/api";
import { ConsentDialog } from "./ConsentDialog";

interface Props {
  onEnrolled: () => void; // triggers the enrolled-list refresh in the parent
}

export function EnrollPanel({ onEnrolled }: Props) {
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [showConsent, setShowConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  function requestEnroll() {
    setError(null);
    setSuccess(null);
    if (!name.trim()) return setError("Enter a name before enrolling.");
    if (!file) return setError("Choose a photo with exactly one face before enrolling.");
    setShowConsent(true);
  }

  async function confirmEnroll() {
    setShowConsent(false);
    if (!file) return;
    setBusy(true);
    try {
      const result = await enrollFace(file, name.trim(), true);
      setSuccess(`Enrolled "${result.name}".`);
      setName("");
      setFile(null);
      if (fileInput.current) fileInput.current.value = "";
      onEnrolled();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Enrollment failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <p className="panel-title">Enroll a face</p>
      {error && <div className="error-banner">{error}</div>}
      {success && <div className="result-meta" style={{ color: "var(--accent)" }}>{success}</div>}

      <div className="field-row">
        <span className="field-label">Name</span>
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Alex Rivera" />
      </div>
      <div className="field-row">
        <span className="field-label">Photo — one face, clearly visible</span>
        <input ref={fileInput} type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </div>

      <div className="btn-row">
        <button className="primary" disabled={busy} onClick={requestEnroll}>
          {busy ? "Enrolling…" : "Enroll"}
        </button>
      </div>

      {showConsent && (
        <ConsentDialog name={name} onConfirm={confirmEnroll} onCancel={() => setShowConsent(false)} />
      )}
    </div>
  );
}
