import { useEffect, useState, useCallback } from "react";
import type { EnrolledEntry } from "../types/api";
import { listFaces, deleteFace, ApiError } from "../services/api";

export interface EnrolledListHandle {
  refresh: () => void;
}

interface Props {
  refreshKey: number; // bumped by the parent whenever an enrollment just succeeded
}

export function EnrolledList({ refreshKey }: Props) {
  const [entries, setEntries] = useState<EnrolledEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listFaces()
      .then(setEntries)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Could not load enrolled faces."));
  }, []);

  useEffect(refresh, [refresh, refreshKey]);

  async function handleDelete(faceId: string) {
    setDeletingId(faceId);
    try {
      await deleteFace(faceId);
      refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Delete failed.");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="panel">
      <p className="panel-title">Enrolled ({entries?.length ?? 0})</p>
      {error && <div className="error-banner">{error}</div>}
      {entries === null && <div className="empty-state">Loading…</div>}
      {entries?.length === 0 && <div className="empty-state">Nobody enrolled yet.</div>}
      {entries?.map((e) => (
        <div className="enrolled-item" key={e.face_id}>
          <span>
            <span className="name">{e.name}</span>{" "}
            <span className="meta">{e.model} · {new Date(e.enrolled_at * 1000).toLocaleDateString()}</span>
          </span>
          <button className="danger" disabled={deletingId === e.face_id} onClick={() => handleDelete(e.face_id)}>
            {deletingId === e.face_id ? "…" : "Remove"}
          </button>
        </div>
      ))}
    </div>
  );
}
