"""Regenerate tests/results/measurements.json and face_roi.png from the fixtures.

Run from the repository root after changing PATCHES:
    python tests/make_results.py
Then look at tests/results/face_roi.png before committing.
"""
import json
from pathlib import Path
import sys

import cv2
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roi_smoke_test import rgb_patch_stats, roi_stats  # noqa: E402

TESTS = Path(__file__).parent
MEDICAL = ("ISIC_0000000.jpg", "ISIC_0000001.jpg", "ISIC_0000002.jpg")
SCALE = 1.5   # preview magnification


def load(name):
    bgr = cv2.imread(str(TESTS / "fixtures" / name))
    if bgr is None:
        raise SystemExit(f"Cannot decode {name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def entry(image, roi, stats):
    rect, means, pixels = stats
    return {"image": image, "roi": roi, "rect": list(rect), "pixels": pixels,
            "rgb_mean": means.tolist()}


def main():
    rgb = load("astronaut.png")
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         min_detection_confidence=0.5) as mesh:
        faces = mesh.process(rgb).multi_face_landmarks
    if not faces:
        raise SystemExit("No face found in astronaut.png")
    patches = roi_stats(rgb, faces[0].landmark)
    rows = [entry("astronaut.png", name, stats) for name, stats in patches.items()]
    for name in MEDICAL:
        image = load(name)
        h, w = image.shape[:2]
        rows.append(entry(name, "central_selected_rectangle",
                          rgb_patch_stats(image, (w // 4, h // 4, 3 * w // 4, 3 * h // 4))))
    (TESTS / "results" / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
    preview = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=SCALE, fy=SCALE,
                         interpolation=cv2.INTER_CUBIC)
    for (x0, y0, x1, y1), _, _ in patches.values():
        cv2.rectangle(preview, (round(x0 * SCALE), round(y0 * SCALE)),
                      (round(x1 * SCALE) - 1, round(y1 * SCALE) - 1), (0, 255, 0), 1)
    cv2.imwrite(str(TESTS / "results" / "face_roi.png"), preview)
    print(json.dumps(rows[:len(patches)], indent=2))


if __name__ == "__main__":
    main()
