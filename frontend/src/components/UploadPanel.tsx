import { useRef, useState } from "react";
import type { AnalyzeOptions } from "../types/api";
import { analyzeImage, analyzeVideo, ApiError } from "../services/api";
import type { AnalyzeImageResponse, AnalyzeVideoResponse } from "../types/api";

interface Props {
  options: AnalyzeOptions;
  onImageResult: (result: AnalyzeImageResponse | null, previewUrl: string | null) => void;
  onVideoResult: (result: AnalyzeVideoResponse | null) => void;
}

export function UploadPanel({ options, onImageResult, onVideoResult }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"image" | "video">("image");
  const fileInput = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setBusy(true);
    setError(null);
    try {
      if (mode === "image") {
        const previewUrl = URL.createObjectURL(file);
        const result = await analyzeImage(file, options);
        onImageResult(result, previewUrl);
      } else {
        onImageResult(null, null);
        const result = await analyzeVideo(file, options);
        onVideoResult(result);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Analysis failed — see the browser console for details.");
      console.error(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <p className="panel-title">
        Analyze
        <span>
          <button
            className={mode === "image" ? "primary" : ""}
            style={{ marginRight: 4, padding: "2px 8px", fontSize: 11 }}
            onClick={() => setMode("image")}
          >
            image
          </button>
          <button
            className={mode === "video" ? "primary" : ""}
            style={{ padding: "2px 8px", fontSize: 11 }}
            onClick={() => setMode("video")}
          >
            video
          </button>
        </span>
      </p>

      {error && <div className="error-banner">{error}</div>}

      <input
        ref={fileInput}
        type="file"
        accept={mode === "image" ? "image/*" : "video/*"}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />

      <div className="btn-row">
        <button className="primary" disabled={busy} onClick={() => fileInput.current?.click()}>
          {busy ? "Analyzing…" : `Choose ${mode} file`}
        </button>
      </div>
    </div>
  );
}
