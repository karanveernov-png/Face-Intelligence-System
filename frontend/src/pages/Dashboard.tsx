import { useState } from "react";
import { TopBar } from "../components/TopBar";
import { SettingsPanel } from "../components/SettingsPanel";
import { UploadPanel } from "../components/UploadPanel";
import { WebcamPanel } from "../components/WebcamPanel";
import { AnnotatedCanvas } from "../components/AnnotatedCanvas";
import { ResultsPanel } from "../components/ResultsPanel";
import { EnrollPanel } from "../components/EnrollPanel";
import { EnrolledList } from "../components/EnrolledList";
import { getModelsStatus } from "../services/api";
import type {
  AnalyzeOptions, AnalyzeImageResponse, AnalyzeVideoResponse, ModelStatus,
} from "../types/api";

type Tab = "analyze" | "webcam" | "enroll";

const DEFAULT_OPTIONS: AnalyzeOptions = {
  run_recognition: true,
  run_expression: true,
  run_age: false,
};

export function Dashboard() {
  const [tab, setTab] = useState<Tab>("analyze");
  const [options, setOptions] = useState<AnalyzeOptions>(DEFAULT_OPTIONS);
  const [models, setModels] = useState<ModelStatus[] | null>(null);
  const [checkingModels, setCheckingModels] = useState(false);
  const [modelCheckError, setModelCheckError] = useState<string | null>(null);

  const [imageResult, setImageResult] = useState<AnalyzeImageResponse | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [videoResult, setVideoResult] = useState<AnalyzeVideoResponse | null>(null);
  const [liveVideoEl, setLiveVideoEl] = useState<HTMLVideoElement | null>(null);

  const [enrollRefreshKey, setEnrollRefreshKey] = useState(0);

  async function checkModels() {
    setCheckingModels(true);
    setModelCheckError(null);
    try {
      setModels(await getModelsStatus());
    } catch (error) {
      setModels(null);
      setModelCheckError(
        error instanceof Error ? error.message : "Could not check model status."
      );
    } finally {
      setCheckingModels(false);
    }
  }
  // onResult is shared by both the upload flow and the webcam stream, so
  // whichever one most recently produced a result is what's displayed.
  const activeFaces = imageResult?.faces ?? [];

  return (
    <div className="app-shell">
      <TopBar
        models={models}
        checkingModels={checkingModels}
        modelCheckError={modelCheckError}
        onCheckModels={checkModels}
      />
      <div className="container">
        <div className="tabs">
          <button className={`tab ${tab === "analyze" ? "active" : ""}`} onClick={() => setTab("analyze")}>Analyze</button>
          <button className={`tab ${tab === "webcam" ? "active" : ""}`} onClick={() => setTab("webcam")}>Webcam</button>
          <button className={`tab ${tab === "enroll" ? "active" : ""}`} onClick={() => setTab("enroll")}>Enroll</button>
        </div>

        <div className="workspace">
          <div>
            <AnnotatedCanvas
              imageUrl={liveVideoEl ? null : imagePreview}
              faces={activeFaces}
              videoEl={liveVideoEl}
            />
            {videoResult && (
              <div className="latency-strip">
                video: {videoResult.frames_processed}/{videoResult.frames_in_file_seen} frames processed
                (every {videoResult.frame_skip}) · {videoResult.processing_fps ?? "—"} fps processed
                · source {videoResult.source_fps} fps
              </div>
            )}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
            {tab === "analyze" && (
              <UploadPanel
                options={options}
                onImageResult={(result, preview) => { setImageResult(result); setImagePreview(preview); setVideoResult(null); }}
                onVideoResult={setVideoResult}
              />
            )}
            {tab === "webcam" && (
              <WebcamPanel
                options={options}
                onResult={setImageResult}
                onVideoElementChange={setLiveVideoEl}
              />
            )}
            {tab === "enroll" && (
              <>
                <EnrollPanel onEnrolled={() => setEnrollRefreshKey((k) => k + 1)} />
                <EnrolledList refreshKey={enrollRefreshKey} />
              </>
            )}

            <SettingsPanel options={options} onChange={setOptions} />
            {tab !== "enroll" && <ResultsPanel result={imageResult} />}
          </div>
        </div>
      </div>
    </div>
  );
}
