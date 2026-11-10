import { useEffect, useRef, useState } from "react";
import type { AnalyzeOptions, AnalyzeImageResponse } from "../types/api";
import { WebcamSession } from "../services/webcam";

interface Props {
  options: AnalyzeOptions;
  onResult: (result: AnalyzeImageResponse | null) => void;
  onVideoElementChange: (el: HTMLVideoElement | null) => void;
}

type SessionState = "idle" | "requesting" | "live" | "stopped";

/**
 * Explicit, visible permission + recording-state handling, per the master
 * spec's "webcam mode, with explicit permission" requirement:
 *   - The video element only exists (and the browser's own camera prompt
 *     only fires) after the person clicks "Start webcam" — never on page load.
 *   - A pulsing on-screen "LIVE" badge (rendered in AnnotatedCanvas, driven
 *     by `onVideoElementChange` here) is visible for the entire time frames
 *     are actually being sent to the server, and disappears the instant
 *     `stop()` is called.
 *   - "Stop webcam" immediately releases the camera track AND closes the
 *     WebSocket — see WebcamSession.stop().
 */
export function WebcamPanel({ options, onResult, onVideoElementChange }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const sessionRef = useRef<WebcamSession | null>(null);
  const [state, setState] = useState<SessionState>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => sessionRef.current?.stop(); // stop the camera if the user navigates away mid-session
  }, []);

  useEffect(() => {
    sessionRef.current?.updateOptions(options);
  }, [options]);

  async function start() {
    setError(null);
    if (!videoRef.current) return;
    const session = new WebcamSession(videoRef.current);
    session.onStateChange = setState;
    session.onResult = (msg) => {
      if ("error" in msg) return; // per-frame errors surface via onError below, not here
      onResult(msg as AnalyzeImageResponse);
    };
    session.onError = (message) => setError(message);
    sessionRef.current = session;
    try {
      await session.start(options);
      onVideoElementChange(videoRef.current);
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Could not access the webcam. Check your browser's camera permission for this site."
      );
      setState("idle");
    }
  }

  function stop() {
    sessionRef.current?.stop();
    sessionRef.current = null;
    onVideoElementChange(null);
    onResult(null);
  }

  return (
    <div className="panel">
      <p className="panel-title">Webcam</p>
      {error && <div className="error-banner">{error}</div>}
      <video ref={videoRef} style={{ display: "none" }} playsInline muted />
      <div className="btn-row">
        {state === "idle" || state === "stopped" ? (
          <button className="primary" onClick={start}>Start webcam</button>
        ) : (
          <button className="danger" onClick={stop} disabled={state === "requesting"}>
            {state === "requesting" ? "Requesting camera access…" : "Stop webcam"}
          </button>
        )}
      </div>
      <p className="caveat">
        Nothing is sent anywhere until you click "Start webcam", which triggers your
        browser's own camera-permission prompt. Frames are analyzed live and never
        saved unless you separately enroll a face.
      </p>
    </div>
  );
}
