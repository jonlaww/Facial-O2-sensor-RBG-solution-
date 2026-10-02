"""Record face ROI colors for pipeline verification; no physiological estimate."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

# Candidate skin patches for a frontal face. Verify their placement visually.
PATCHES = {"forehead": (107, 9, 336, 151),
           "cheek_a": (50, 101, 205, 187),
           "cheek_b": (280, 330, 425, 411)}


def rgb_patch_stats(rgb, rect):
    """Sample an explicitly supplied rectangle; coordinates do not imply skin."""
    height, width = rgb.shape[:2]
    if not np.isfinite(rect).all():
        return None
    x0, y0, x1, y1 = rect
    x0, y0 = int(np.floor(x0)), int(np.floor(y0))
    x1, y1 = int(np.ceil(x1)), int(np.ceil(y1))
    x0, x1 = np.clip([x0, x1], 0, width)
    y0, y1 = np.clip([y0, y1], 0, height)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = rgb[y0:y1, x0:x1]
    means = patch.mean(axis=(0, 1))
    if not np.isfinite(means).all():
        return None
    return ((int(x0), int(y0), int(x1), int(y1)), means,
            int(patch.shape[0] * patch.shape[1]))


def roi_stats(rgb, landmarks):
    """Return clipped rectangles and RGB means; reject nonfinite/empty patches."""
    height, width = rgb.shape[:2]
    result = {}
    for name, indices in PATCHES.items():
        points = np.array([(landmarks[i].x * width, landmarks[i].y * height)
                           for i in indices])
        if not np.isfinite(points).all():
            continue
        x0, y0 = np.floor(points.min(axis=0)).astype(int)
        x1, y1 = np.ceil(points.max(axis=0)).astype(int)
        stats = rgb_patch_stats(rgb, (x0, y0, x1, y1))
        if stats is not None:
            result[name] = stats
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, help="Use a local video instead of a camera")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if not hasattr(mp, "solutions"):
        parser.error("Legacy FaceMesh unavailable. Install requirements-smoke.txt.")
    args.output.mkdir(parents=True, exist_ok=True)
    run = args.output / time.strftime("roi-%Y%m%d-%H%M%S")
    csv_path = run.with_suffix(".csv")
    if csv_path.exists():
        parser.error("Run filename already exists; retry with a different output folder")
    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        cap.release()
        raise SystemExit("Cannot open camera/video. Check permissions, camera index, or path.")
    frames = face_frames = complete_frames = rows = 0
    start = time.monotonic()
    source_start = None
    status = "duration_complete"
    try:
        with csv_path.open("x", newline="") as out, mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as mesh:
            writer = csv.writer(out)
            writer.writerow(["frame", "elapsed_s", "capture_unix_s", "source_ms", "roi",
                             "x0", "y0", "x1", "y1", "pixels", "red", "green", "blue"])
            while True:
                if not args.video and time.monotonic() - start >= args.seconds:
                    break
                ok, bgr = cap.read()
                if not ok:
                    status = "video_end" if args.video else "camera_read_failed"
                    break
                source_ms = cap.get(cv2.CAP_PROP_POS_MSEC) if args.video else None
                if args.video:
                    if source_start is None:
                        source_start = source_ms
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    elapsed = ((source_ms - source_start) / 1000 if source_ms > source_start
                               else frames / fps if fps > 0 else 0)
                    if elapsed >= args.seconds:
                        break
                else:
                    elapsed = time.monotonic() - start
                capture_unix = time.time()
                frames += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                detected = mesh.process(rgb)
                patches = {}
                if detected.multi_face_landmarks:
                    face_frames += 1
                    patches = roi_stats(rgb, detected.multi_face_landmarks[0].landmark)
                    complete_frames += len(patches) == len(PATCHES)
                for name, (rect, means, pixels) in patches.items():
                    writer.writerow([frames, elapsed, capture_unix, source_ms, name,
                                     *rect, pixels, *means.tolist()])
                    rows += 1
                    x0, y0, x1, y1 = rect
                    cv2.rectangle(bgr, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)
                    cv2.putText(bgr, name, (x0, max(12, y0 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(bgr, f"ROI test {elapsed:.1f}s | face: {bool(patches)}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                if not args.headless:
                    cv2.imshow("Face ROI smoke test", bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        status = "user_stopped"
                        break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    summary = {"status": status, "frames": frames, "face_frames": face_frames,
               "frames_with_all_rois": complete_frames, "csv_rows": rows,
               "requested_seconds": args.seconds, "source": "video" if args.video else "camera",
               "versions": {name: importlib.metadata.version(name) for name in
                            ("mediapipe", "numpy", "scipy", "opencv-contrib-python")},
               "interpretation": "Color traces only. Visual ROI placement still requires review."}
    run.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Recorded: {csv_path}")
    if rows == 0 or status == "camera_read_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
