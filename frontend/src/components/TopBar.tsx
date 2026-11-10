import type { ModelStatus } from "../types/api";

interface Props {
  models: ModelStatus[] | null;
  checkingModels: boolean;
  modelCheckError: string | null;
  onCheckModels: () => void;
}

export function TopBar({ models, checkingModels, modelCheckError, onCheckModels }: Props) {
  return (
    <header className="topbar">
      <div className="container topbar-inner">
        <div className="brand">
          <span className="brand-mark">FI</span>
          <span className="brand-name">Face Intelligence</span>
        </div>
        <div className="topbar-actions">
          {models && (
            <div className="model-dots" aria-label="Model test results">
              {models.map((m) => (
                <span className="model-dot" key={m.name} title={m.error ?? `${m.name}: ${m.purpose}`}>
                  <span className={`dot ${m.loaded ? "ok" : "bad"}`} />
                  {m.name}
                </span>
              ))}
            </div>
          )}
          <button type="button" onClick={onCheckModels} disabled={checkingModels}>
            {checkingModels ? "Testing…" : "Test models"}
          </button>
        </div>
      </div>
      {modelCheckError && (
        <div className="container model-check-error" role="alert">
          Model check failed: {modelCheckError}
        </div>
      )}
    </header>
  );
}
