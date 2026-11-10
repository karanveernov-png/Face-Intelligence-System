import type { AnalyzeOptions, WsServerMessage } from "../types/api";

/**
 * Owns the three things "webcam mode" actually needs, kept separate from any
 * React component so they're independently testable/reasoned about:
 *   1. requesting camera permission and owning the MediaStream,
 *   2. periodically grabbing a JPEG frame off that stream via a canvas,
 *   3. sending each frame over the WebSocket protocol verified in
 *      backend/app/api/ws.py and backend/tests/test_ws.py, and delivering
 *      results back to the caller.
 *
 * This class never assumes the backend's analysis keeps up with real camera
 * frame rate (it doesn't, on CPU-only hardware — see docs/OPENCV_FUNDAMENTALS.md
 * section 15). It sends one frame, waits for that frame's result, then sends
 * the next -- a simple backpressure strategy that can't pile up a queue of
 * stale frames if the backend is slower than the camera.
 */
export class WebcamSession {
  private stream: MediaStream | null = null;
  private socket: WebSocket | null = null;
  private video: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private running = false;
  private waitingForResult = false;
  private options: AnalyzeOptions | null = null;
  private optionsVersion = 0;
  private requestOptionsVersion = 0;

  onResult: (msg: WsServerMessage) => void = () => {};
  onError: (message: string) => void = () => {};
  onStateChange: (state: "idle" | "requesting" | "live" | "stopped") => void = () => {};

  constructor(videoEl: HTMLVideoElement) {
    this.video = videoEl;
    this.canvas = document.createElement("canvas");
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable in this browser.");
    this.ctx = ctx;
  }

  async start(options: AnalyzeOptions): Promise<void> {
    this.options = { ...options };
    this.onStateChange("requesting");
    // This call is what triggers the browser's own native permission prompt --
    // there is no way for this app to show camera video without the person
    // explicitly granting it through that OS/browser-level dialog.
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();

    this.canvas.width = this.video.videoWidth || 640;
    this.canvas.height = this.video.videoHeight || 480;

    const params = new URLSearchParams(this.serializeOptions(this.options));
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    this.socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/analyze?${params}`);

    await new Promise<void>((resolve, reject) => {
      if (!this.socket) return reject(new Error("socket not created"));
      this.socket.onopen = () => resolve();
      this.socket.onerror = () => reject(new Error("Could not connect to the analysis server."));
    });

    this.sendOptions();
    this.socket.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data) as WsServerMessage;
        if ("type" in msg && msg.type === "options_applied") return;
        this.waitingForResult = false;
        if (this.requestOptionsVersion !== this.optionsVersion) return;
        if ("error" in msg) this.onError(msg.error);
        else this.onResult(msg);
      } catch {
        this.onError("Received a malformed response from the analysis server.");
      }
    };
    this.socket.onclose = () => {
      this.running = false;
      this.onStateChange("stopped");
    };

    this.running = true;
    this.onStateChange("live");
    this.sendLoop();
  }

  updateOptions(options: AnalyzeOptions): void {
    this.options = { ...options };
    this.optionsVersion += 1;
    this.sendOptions();
  }

  private serializeOptions(options: AnalyzeOptions): Record<string, string> {
    return {
      run_recognition: String(options.run_recognition),
      run_expression: String(options.run_expression),
      run_age: String(options.run_age),
    };
  }

  private sendOptions(): void {
    if (this.socket?.readyState !== WebSocket.OPEN || !this.options) return;
    this.socket.send(JSON.stringify({
      type: "options",
      ...this.options,
    }));
  }

  private sendLoop(): void {
    const tick = () => {
      if (!this.running) return;
      if (!this.waitingForResult && this.socket?.readyState === WebSocket.OPEN) {
        this.waitingForResult = true;
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        this.canvas.toBlob(
          (blob) => {
            if (!blob || this.socket?.readyState !== WebSocket.OPEN) {
              this.waitingForResult = false;
              return;
            }
            blob.arrayBuffer()
              .then((buf) => {
                if (this.socket?.readyState !== WebSocket.OPEN) {
                  this.waitingForResult = false;
                  return;
                }
                this.requestOptionsVersion = this.optionsVersion;
                this.socket.send(buf);
              })
              .catch((error: unknown) => {
                this.waitingForResult = false;
                this.onError(error instanceof Error ? error.message : "Could not encode webcam frame.");
              });
          },
          "image/jpeg",
          0.85
        );
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    this.socket?.close();
    this.socket = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.onStateChange("idle");
  }

  get isLive(): boolean {
    return this.running;
  }
}
