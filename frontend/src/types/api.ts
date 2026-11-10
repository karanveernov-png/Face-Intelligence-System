// Mirrors backend/app/schemas/models.py and the to_dict() shapes in
// backend/app/services/*.py. Kept in one file, by hand, rather than
// generated — small enough surface that a generator would add more
// ceremony than it saves. If the backend's shapes drift, this is the
// file to update alongside them.

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Landmarks {
  left_eye: [number, number] | null;
  right_eye: [number, number] | null;
  nose: [number, number] | null;
  mouth_left: [number, number] | null;
  mouth_right: [number, number] | null;
}

export interface IdentityResult {
  identity: string; // "Unknown" when unmatched
  matched: boolean;
  distance: number | null;
  threshold: number;
  note: string;
}

export interface ExpressionResult {
  label: string; // one of 8 RAF-DB categories (or 7 FER categories), or "uncertain"
  certain: boolean;
  scores: Record<string, number>;
  note: string;
}

export interface AgeResult {
  enabled: boolean;
  estimated_age: number | null;
  estimated_age_range: string | null; // e.g. "25-32" — a presentation bucket, see backend/app/services/age_estimation.py
  estimated_age_uncertainty_years?: number | null; // honest disagreement-based uncertainty from the 2-model ensemble; absent/null when no estimate could be made
  gender?: string | null;
  gender_confidence?: number | null;
  note: string;
}

export interface FaceResult {
  face_index: number;
  bbox: BBox;
  confidence: number;
  detector_backend: string;
  landmarks: Landmarks | null;
  identity?: IdentityResult | null;
  expression?: ExpressionResult | null;
  age?: AgeResult | null;
}

export interface DetectorMeta {
  fallback_used: boolean;
  errors: string[];
  backend_used: string;
  latency_ms: number;
}

export interface AnalyzeImageResponse {
  face_count: number;
  faces: FaceResult[];
  detector: DetectorMeta;
  total_latency_ms: number;
  task_errors: string[];
}

export interface VideoFrameResult {
  frame_index: number;
  face_count: number;
  faces: FaceResult[];
  active_tracks: number;
  detector_backend?: string;
}

export interface AnalyzeVideoResponse {
  source_fps: number;
  frames_in_file_seen: number;
  frames_processed: number;
  frame_skip: number;
  detection_interval?: number;
  processing_fps: number | null;
  results: VideoFrameResult[];
}

export interface EnrollResponse {
  face_id: string;
  name: string;
  retained_image: boolean;
}

export interface EnrolledEntry {
  face_id: string;
  name: string;
  model: string;
  enrolled_at: number;
}

export interface ModelStatus {
  name: string;
  purpose: string;
  loaded: boolean;
  backend: string;
  error: string | null;
}

export interface AnalyzeOptions {
  run_recognition: boolean;
  run_expression: boolean;
  run_age: boolean;
}

export interface ApiErrorBody {
  detail: string;
}

// WebSocket webcam-streaming protocol (see backend/app/api/ws.py).
// Client -> server: one binary JPEG frame per message.
// Server -> client: one JSON AnalyzeImageResponse per processed frame, or
// a JSON { error: string } if that single frame failed to process (the
// connection itself stays open — one bad frame doesn't end the session).
export type WsServerMessage =
  | AnalyzeImageResponse
  | { error: string }
  | { type: "options_applied" };
