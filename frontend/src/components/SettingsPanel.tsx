import type { AnalyzeOptions } from "../types/api";

interface Props {
  options: AnalyzeOptions;
  onChange: (next: AnalyzeOptions) => void;
}

const ROWS: { key: keyof AnalyzeOptions; label: string; hint: string }[] = [
  { key: "run_recognition", label: "Identity matching (ArcFace-R50 INT8)", hint: "5-point Umeyama aligned 512-D match against InsightFace's w600k_r50 backbone (~99.8% LFW per InsightFace's published figures. ~42 MB INT8)." },
  { key: "run_expression", label: "Expression analysis (HSEmotion ENet-B0 INT8)", hint: "8 RAF-DB classes + temporal EMA smoothing (86.9% RAF-DB. ~4.0 MB INT8)." },
  { key: "run_age", label: "Age + gender estimation (GenderAge + FairFace ensemble)", hint: "Averages two independent open models and reports their disagreement as an honest uncertainty range, instead of a single confident number. Approximate for every face; never used to determine whether someone is a minor." },
];

export function SettingsPanel({ options, onChange }: Props) {
  return (
    <div className="panel">
      <p className="panel-title">Analysis tasks (Accuracy-First INT8 Stack)</p>
      <p className="settings-note">
        Detection runs SCRFD-10G INT8 (~4.3 MB, 95.16% WiderFace Hard) with decoupled optical tracking.
      </p>
      {ROWS.map((row) => (
        <div className="toggle-row" key={row.key}>
          <span>
            {row.label}
            <span className="toggle-hint">{row.hint}</span>
          </span>
          <button
            type="button"
            className={`task-switch ${options[row.key] ? "enabled" : ""}`}
            role="switch"
            aria-checked={options[row.key]}
            aria-label={`${row.label}: ${options[row.key] ? "enabled" : "disabled"}`}
            onClick={() => onChange({ ...options, [row.key]: !options[row.key] })}
          >
            <span className="task-switch-state">{options[row.key] ? "On" : "Off"}</span>
            <span className="task-switch-track" aria-hidden="true">
              <span className="task-switch-thumb" />
            </span>
          </button>
        </div>
      ))}
    </div>
  );
}
