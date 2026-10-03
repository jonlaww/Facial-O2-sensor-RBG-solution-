"""Ratio-of-ratios SpO2 baseline on the open UW finger-camera hypoxemia dataset.

Data: Hoffman et al., npj Digital Medicine 5, 146 (2022), MIT license,
https://github.com/ubicomplab/oximetry-phone-cam-data (not bundled here).
Input is per-frame mean R,G,B at 30 Hz: the same shape roi_smoke_test.py logs.
This is contact finger PPG with the phone flash, NOT facial video. It evaluates
this baseline against reference pulse oximeters, not a webcam pointed at a face.

Usage:
    git clone https://github.com/ubicomplab/oximetry-phone-cam-data.git
    git -C oximetry-phone-cam-data checkout c483ae8
    python validation/uw_ratio_of_ratios.py oximetry-phone-cam-data
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import signal

SUBJECTS = ["100001", "100002", "100003", "100004", "100005", "100006"]
FPS = 30
WINDOW_S = 10
BAND_HZ = (0.7, 3.5)
REFERENCE_COLUMNS = ("SpO2 1", "SpO2 2", "SpO2 4", "SpO2 5")


def load_reference(path):
    """Per-second mean of the clinical oximeters that report a plausible value."""
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            values = []
            for column in REFERENCE_COLUMNS:
                try:
                    value = float(row[column])
                except (KeyError, TypeError, ValueError):
                    continue
                if 50 <= value <= 100:
                    values.append(value)
            rows.append(np.mean(values) if values else np.nan)
    return np.array(rows)


def windows(rgb, reference):
    """Yield (features, reference SpO2, reference spread) per 10 s window."""
    sos = signal.butter(3, BAND_HZ, btype="bandpass", fs=FPS, output="sos")
    ac_all = signal.sosfiltfilt(sos, rgb, axis=0)
    n = WINDOW_S * FPS
    for k in range(min(len(rgb) // n, len(reference) // WINDOW_S)):
        ref = reference[k * WINDOW_S:(k + 1) * WINDOW_S]
        if not np.isfinite(ref).all():
            continue
        dc = rgb[k * n:(k + 1) * n].mean(axis=0)
        ac = ac_all[k * n:(k + 1) * n].std(axis=0)
        if (dc <= 1).any() or (dc >= 254).any() or (ac <= 0).any():
            continue
        perfusion = ac / dc
        yield (np.array([perfusion[0] / perfusion[2], perfusion[0] / perfusion[1]]),
               ref.mean(), ref.max() - ref.min())


def fit_predict(x_train, y_train, x_test):
    design = np.column_stack([np.ones(len(x_train)), x_train])
    coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return np.column_stack([np.ones(len(x_test)), x_test]) @ coef


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-ref-spread", type=float, default=3.0,
                        help="skip windows where reference SpO2 moves more than this (%%)")
    args = parser.parse_args()
    data = {}
    for subject in SUBJECTS:
        reference = load_reference(args.dataset / "data" / "gt" / f"{subject}.csv")
        feats, target = [], []
        for hand in ("Left", "Right"):
            rgb = np.genfromtxt(args.dataset / "data" / "ppg-csv" / hand / f"{subject}.csv",
                                delimiter=",", skip_header=1)
            for x, y, spread in windows(rgb, reference):
                if spread <= args.max_ref_spread and np.isfinite(x).all():
                    feats.append(x)
                    target.append(y)
        data[subject] = (np.array(feats), np.array(target))
    report = {"windows": {s: len(data[s][1]) for s in SUBJECTS}, "features": {}}
    truth = np.concatenate([data[s][1] for s in SUBJECTS])
    report["reference_spo2_range"] = [float(truth.min()), float(truth.max())]
    report["windows_below_90"] = int((truth < 90).sum())
    for name, column in (("red_over_blue", 0), ("red_over_green", 1)):
        predicted, constant = [], []
        for held_out in SUBJECTS:
            train = [s for s in SUBJECTS if s != held_out]
            x_train = np.concatenate([data[s][0][:, column] for s in train])
            y_train = np.concatenate([data[s][1] for s in train])
            predicted.append(fit_predict(x_train, y_train, data[held_out][0][:, column]))
            constant.append(np.full(len(data[held_out][1]), y_train.mean()))
        predicted, constant = np.concatenate(predicted), np.concatenate(constant)
        report["features"][name] = {
            "rmse_model": rmse(predicted, truth),
            "rmse_constant_predictor": rmse(constant, truth),
            "pearson_r": float(np.corrcoef(predicted, truth)[0, 1]),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
