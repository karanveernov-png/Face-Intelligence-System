import { useState } from "react";

interface Props {
  name: string;
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * This is the actual "consent collection" step the backend's
 * `consent_confirmed` flag assumes already happened (see docs/PRIVACY.md —
 * the API can refuse to proceed without the flag, but it cannot verify
 * real-world informed consent itself). The checkbox must be explicitly
 * ticked; the confirm button stays disabled until it is.
 */
export function ConsentDialog({ name, onConfirm, onCancel }: Props) {
  const [checked, setChecked] = useState(false);

  return (
    <div className="consent-overlay" role="dialog" aria-modal="true" aria-labelledby="consent-title">
      <div className="consent-card">
        <h3 id="consent-title">Confirm enrollment consent</h3>
        <p>
          You're about to enroll <strong>{name || "this person"}</strong> in the local face
          gallery. Their face embedding (not the photo, by default) will be stored on this
          device and compared against future uploads to recognize them by name.
        </p>
        <p>
          This system never identifies strangers and never enrolls anyone automatically —
          this is the only way a face gets added. Enrollment can be removed at any time
          from the "Enrolled" list.
        </p>
        <label className="consent-check">
          <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} />
          <span>I have the enrolled person's informed consent to store their face data this way.</span>
        </label>
        <div className="btn-row">
          <button className="primary" disabled={!checked} onClick={onConfirm}>Confirm enrollment</button>
          <button onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  );
}
