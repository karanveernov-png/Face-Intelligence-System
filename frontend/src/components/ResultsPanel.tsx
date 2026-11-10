import type { AnalyzeImageResponse, FaceResult } from "../types/api";

interface Props {
  result: AnalyzeImageResponse | null;
}

export function ResultsPanel({ result }: Props) {
  if (!result) {
    return (
      <div className="panel">
        <p className="panel-title">Results</p>
        <div className="empty-state">No analysis yet.</div>
      </div>
    );
  }

  return (
    <div className="panel">
      <p className="panel-title">
        Results
        <span className="mono" style={{ color: "var(--text-faint)", fontSize: 11 }}>
          {result.total_latency_ms.toFixed(0)}ms
        </span>
      </p>
      <div className="result-meta">
        <span>faces: {result.face_count}</span>
        <span>detector: {result.detector.backend_used}</span>
        {result.detector.fallback_used && <span style={{ color: "var(--warn)" }}>fallback active</span>}
      </div>

      {result.face_count === 0 && <div className="empty-state">No faces detected in this frame.</div>}

      {result.faces.map((f) => (
        <FaceCard key={f.face_index} face={f} />
      ))}

      {result.task_errors.length > 0 && (
        <div className="task-errors">
          {result.task_errors.length} task{result.task_errors.length > 1 ? "s" : ""} failed and were skipped for this result:
          <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
            {result.task_errors.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

function FaceCard({ face }: { face: FaceResult }) {
  return (
    <div className="face-card">
      <div className="face-card-head">
        <span className={`identity-name ${face.identity?.matched ? "" : "unknown"}`}>
          {face.identity ? face.identity.identity : "recognition off"}
        </span>
        <span className="idx">#{face.face_index} · conf {face.confidence.toFixed(2)}</span>
      </div>

      {face.identity && (
        <div className="row-line">
          <span>match distance</span>
          <span className="v">{face.identity.distance != null ? face.identity.distance.toFixed(3) : "—"} / {face.identity.threshold}</span>
        </div>
      )}

      {face.expression && (
        <div className="row-line">
          <span>expression</span>
          <span className="v">
            {face.expression.label}{" "}
            <span className={`badge ${face.expression.certain ? "confident" : "uncertain"}`}>
              {face.expression.certain ? "confident" : "uncertain"}
            </span>
          </span>
        </div>
      )}

      {face.age?.enabled && face.age.estimated_age != null && (
        <div className="row-line">
          <span>age</span>
          <span className="v">
            {face.age.estimated_age_range ?? `~${face.age.estimated_age}`} yrs
            <span style={{ color: "var(--text-faint)" }}>
              {" "}(point est. {face.age.estimated_age}
              {face.age.estimated_age_uncertainty_years != null && ` ± ${face.age.estimated_age_uncertainty_years}`})
            </span>
          </span>
        </div>
      )}
      {face.age?.enabled && face.age.estimated_age == null && (
        <div className="row-line">
          <span>age</span>
          <span className="v">could not estimate</span>
        </div>
      )}

      <p className="caveat">
        {face.identity?.note ?? face.expression?.note ?? face.age?.note ?? "A bounding box only — no identity, expression, or age tasks were enabled for this result."}
      </p>
    </div>
  );
}
