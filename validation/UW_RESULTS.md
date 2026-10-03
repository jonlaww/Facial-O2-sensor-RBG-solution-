# UW ratio baseline reproduction

Run on Python 3.12.14 Linux using the supplied script and dataset commit c483ae8fe4cd32b0b9ce412d07299b05279fc58f. Result JSON: `uw_results.json`. Dataset MIT license, not bundled.

Source: Hoffman et al. (2022), Smartphone camera oximetry in an induced hypoxemia study, npj Digital Medicine 5, 146. https://github.com/ubicomplab/oximetry-phone-cam-data

Protocol: average available reference oximeters with values 50–100, assume first samples aligned and constant 30 Hz RGB / 1 Hz reference rates, bandpass RGB 0.7–3.5 Hz with zero-phase filtering, use nonoverlapping 10-second windows, normalize filtered standard deviation AC by mean DC, fit affine saturation versus red/blue or red/green AC/DC ratio on five subjects, evaluate the held-out sixth. Both hands remain in the same subject fold. Retain windows with reference spread at most 3 percentage points. No clipping of predictions.

Retained 1,170 hand-windows across six subjects; 622 have averaged reference below 90%. Averaged reference range 64.8125–99.89%. The repository describes the study range as 70–100%; the lower observed average is reported as found, not independently clinically verified.

| Predictor | Pooled RMSE (percentage points) | Pearson r |
| --- | ---: | ---: |
| Red/blue ratio | 10.2657 | -0.2149 |
| Red/green ratio | 10.3160 | -0.1318 |
| Training-mean constant | 9.0025 | — |

The supplied result is reproduced: both affine ratio baselines are worse than the constant under this protocol. This is a baseline evaluation, not proof that the arithmetic is correct, nor a clinical validation. It does not test the historical 100−5×color heuristic or the facial camera system.

Limitations: four pulse-ox readings are surrogate references, not arterial co-oximetry; different devices may disagree or lag. Timestamp continuity, missing samples, camera exposure/clipping at pixel level and per-device agreement were not audited. The filter uses future samples offline; performance does not represent real-time causal processing. Window selection uses reference outcomes and preferentially removes saturation transitions. Both hands share reference labels, so the 1,170 windows are not independent participants. No uncertainty interval or per-subject accuracy conclusion is claimed.
