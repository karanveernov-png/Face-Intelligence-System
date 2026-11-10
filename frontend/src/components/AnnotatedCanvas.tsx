import { useEffect, useRef } from "react";
import type { FaceResult } from "../types/api";

interface Props {
  imageUrl: string | null;
  faces: FaceResult[];
  videoEl?: HTMLVideoElement | null; // when set (webcam mode), draws the live video frame instead of imageUrl
}

/**
 * Draws whatever image source is active (an uploaded photo, or the live
 * webcam video element) at native resolution onto a canvas, then overlays
 * real detection results on top: a box per face, landmark points, and an
 * inline label. Never draws a box that wasn't actually in `faces` -- no
 * placeholder or illustrative boxes.
 */
export function AnnotatedCanvas({ imageUrl, faces, videoEl }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = (source: CanvasImageSource, w: number, h: number) => {
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(source, 0, 0, w, h);
      drawOverlays(ctx, faces);
    };

    if (videoEl && videoEl.videoWidth > 0) {
      draw(videoEl, videoEl.videoWidth, videoEl.videoHeight);
      return;
    }

    if (imageUrl) {
      const img = new Image();
      img.onload = () => draw(img, img.naturalWidth, img.naturalHeight);
      img.src = imageUrl;
      imgRef.current = img;
    }
  }, [imageUrl, faces, videoEl]);

  // In webcam mode, redraw continuously so the live feed keeps moving even
  // between result updates (results arrive slower than the camera's frame
  // rate -- see docs/OPENCV_FUNDAMENTALS.md section 15).
  useEffect(() => {
    if (!videoEl) return;
    let raf: number;
    const loop = () => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (canvas && ctx && videoEl.videoWidth > 0) {
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        drawOverlays(ctx, faces);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoEl, faces]);

  return (
    <div className="viewfinder">
      {!imageUrl && !videoEl && <div className="viewfinder-empty">No image loaded. Upload a photo, a video, or start the webcam.</div>}
      {videoEl && (
        <span className="live-badge">
          <span className="live-dot" /> LIVE — webcam frames are being analyzed
        </span>
      )}
      <canvas ref={canvasRef} style={{ display: imageUrl || videoEl ? "block" : "none" }} />
      <span className="vf-corner tl" /><span className="vf-corner tr" />
      <span className="vf-corner bl" /><span className="vf-corner br" />
    </div>
  );
}

function drawOverlays(ctx: CanvasRenderingContext2D, faces: FaceResult[]) {
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#5ec8b8";
  for (const f of faces) {
    const { x, y, w, h } = f.bbox;
    ctx.strokeStyle = accent;
    ctx.lineWidth = Math.max(2, Math.round(w * 0.01));
    ctx.strokeRect(x, y, w, h);

    const parts: string[] = [];
    if (f.identity) parts.push(f.identity.identity);
    if (f.expression && f.expression.label !== "uncertain") parts.push(f.expression.label);
    
    // Use exact 5-year interval string sent from backend
    if (f.age?.estimated_age_range) {
      parts.push(f.age.estimated_age_range);
    }
    
    const label = parts.length ? parts.join(" · ") : `face ${f.face_index}`;

    // FIXED: Clean single-line multiplication formula for font scaling
    ctx.font = `${Math.max(12, Math.round(w * 0.06))}px Inter, sans-serif`;
    
    const metrics = ctx.measureText(label);
    const labelH = Math.max(16, Math.round(w * 0.08));
    ctx.fillStyle = accent;
    ctx.fillRect(x, Math.max(0, y - labelH), metrics.width + 10, labelH);
    ctx.fillStyle = "#0d1310";
    ctx.fillText(label, x + 5, Math.max(labelH - 4, y - 5));

    if (f.landmarks) {
      ctx.fillStyle = "#ffffff";
      for (const pt of Object.values(f.landmarks)) {
        if (!pt) continue;
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], Math.max(2, w * 0.01), 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}