# Verification — 2026-10-02

Base: main at f91a7e2c3a1499762310da82aad6db0647d588db.

Passed in isolated Python 3.12.14 Linux environment:
- Installation of MediaPipe 0.10.21, NumPy 1.26.4, SciPy 1.13.1, OpenCV contrib 4.11.0.86.
- Syntax compilation of the smoke test and both historical scripts.
- Synthetic landmark/color checks: constant [210,120,30] RGB preserved; partial rectangles clipped; completely off-image, empty, and nonfinite patches rejected.
- Actual legacy FaceMesh initialization and inference on a blank frame returned no face.
- Headless video CLI on a generated 10 FPS blank video processed exactly 10 frames for a one-second run, wrote CSV header and JSON versions/counts, and exited 1 as expected for no detected ROIs.
- git diff --check.

Not verified: physical webcam access, GUI preview, landmark placement on a real face, real-face color traces, macOS/Windows installation. No physiological accuracy was tested or claimed. Discord assets were not recovered.

The required next check is the 30-second local camera run described in README.md. Review all three boxes visually before using its color traces.
