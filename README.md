# Face ROI smoke test

This repository includes an experimental 2024 oxygen-estimation demo. Its displayed percentages are uncalibrated color heuristics. The supported verification entry point is now `roi_smoke_test.py`, which records color traces only.

## Run a 30-second check

Use Python 3.12 in a fresh environment, separate from your existing CV projects:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-smoke-lock.txt
python roi_smoke_test.py --seconds 30
```

Windows activation: `.venv\Scripts\activate`. Grant camera access and close other apps using the camera. Try `--camera 1` if index 0 is incorrect. Press `q` to stop early.

Alternatively, use a local frontal-face video:

```bash
python roi_smoke_test.py --video face_clip.mp4 --seconds 30
```

Video duration uses source timestamps with frame-rate fallback; camera duration uses a monotonic clock. `--headless` disables the preview for server use.

## What to review

Sit facing the camera in steady light. Verify that the three rectangles lie on forehead and cheek skin, avoiding eyes, nose, mouth, hair and background. These are candidate landmark patches, not validated anatomical segmentations. Turn slightly to see whether placement remains acceptable.

The `runs/` folder receives a CSV of timestamps, rectangles, pixel counts and RGB means, plus a JSON summary with installed package versions and face/ROI counts. A useful smoke result has all three nonempty ROIs on most frontal-face frames, finite RGB traces, and visually correct placement. Detection counts alone do not prove correct placement. A nonzero exit indicates no recorded ROIs or a camera read failure. No face detected on a frame means no CSV rows for that frame; use frame numbers and summary counts to identify gaps.

RGB ordering is explicitly red=0, green=1, blue=2 after BGR-to-RGB conversion. Empty, nonfinite or off-image rectangles are rejected; partial rectangles are clipped. No oxygen percentage is computed.

## Scope and assets

The two original scripts remain historical examples; only their reversed red/blue indexing has been corrected. Their oxygen formulas are not validated and their ROI geometry is not the new smoke-test geometry. The historical weights remain unused.

Issue #1 contains three external Discord URLs. Review on 2026-10-02 returned HTTP 403 for all three; URL expiry fields point to 2024-02-26. This does not prove the original attachments were deleted. No images were recovered, inspected, or bundled. The issue describes generated pseudo-hypoxic faces: if recovered, they can serve as image-pipeline fixtures, not physiological oxygen ground truth. The three repository PDFs remain references.

The full lock records the tested Python 3.12 Linux environment; installation on macOS/Windows remains unverified. `requirements-smoke.txt` lists the four direct dependencies. The dependency file deliberately pins a legacy MediaPipe baseline for this small check. A later migration can use Face Landmarker. SciPy is included because the historical scripts import it. The smoke test itself performs no filtering or AC/DC separation.

Camera data remains local. CSV files contain no saved face images, but still record facial color measurements. Avoid committing recordings; `runs/` is ignored.

## Local verification

See `VERIFICATION.md` for the environment and checks completed before delivery. Actual camera placement must be reviewed on your computer.
