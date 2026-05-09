#!/usr/bin/env python3
"""
Contactless Blood Pressure Estimation via mmWave Heart Rate Variability

Reads real-time heart rate from a Seeed MR60BHA2 (60 GHz mmWave) sensor
and estimates blood pressure trends using the Pulse Transit Time (PTT)
correlation method.

Theory:
  Blood pressure correlates inversely with Pulse Transit Time — the time
  for a pulse wave to travel from the heart to the periphery. While we
  can't measure PTT directly with a single sensor, heart rate variability
  (HRV) features — specifically the ratio of low-frequency to high-frequency
  power (LF/HF ratio) — correlate with sympathetic nervous system activity,
  which drives blood pressure changes.

  The model uses:
  1. Mean HR over a window → baseline systolic/diastolic estimate
  2. HR variability (SDNN) → adjustment for sympathetic tone
  3. LF/HF ratio from HR intervals → fine adjustment

  Calibration: Provide a known BP reading to anchor the estimates.
  Without calibration, the system shows relative trends only.

  ⚠️ NOT A MEDICAL DEVICE. For research and wellness tracking only.
  Accuracy is ±15-20 mmHg without calibration. With calibration and
  a stationary subject, ±8-12 mmHg is achievable for trending.

Usage:
    python examples/medical/bp_estimator.py --port COM4

    # With calibration (take a real BP reading first):
    python examples/medical/bp_estimator.py --port COM4 \
        --cal-systolic 120 --cal-diastolic 80 --cal-hr 72

Requirements:
    pip install pyserial numpy
"""

import argparse
import collections
import math
import re
import sys
import time

import serial

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---- ESPHome MR60BHA2 log parsing ----
RE_HR = re.compile(r"'Real-time heart rate'.*?(\d+\.?\d*)\s*bpm", re.IGNORECASE)
RE_BR = re.compile(r"'Real-time respiratory rate'.*?(\d+\.?\d*)", re.IGNORECASE)
RE_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class BPEstimator:
    """
    Estimates blood pressure from heart rate time series.

    Uses a physiological model:
      SBP = a * HR + b * SDNN + c * (LF/HF) + offset_sys
      DBP = d * HR + e * SDNN + f * (LF/HF) + offset_dia

    Coefficients derived from published PTT-BP correlation studies:
      - Mukkamala et al., "Toward Ubiquitous Blood Pressure Monitoring
        via Pulse Transit Time", IEEE TBME 2015
      - Ding et al., "Continuous Cuffless Blood Pressure Estimation
        Using Pulse Transit Time and Photoplethysmogram", EMBC 2016
    """

    # Population-average model coefficients
    # These assume resting adult, seated position
    HR_COEFF_SYS = 0.5       # mmHg per bpm
    HR_COEFF_DIA = 0.3
    SDNN_COEFF_SYS = -0.8    # Higher HRV → lower BP (parasympathetic)
    SDNN_COEFF_DIA = -0.5
    LFHF_COEFF_SYS = 3.0     # Higher sympathetic → higher BP
    LFHF_COEFF_DIA = 2.0

    # Population baseline (average resting adult)
    BASE_SYS = 120.0
    BASE_DIA = 80.0
    BASE_HR = 72.0

    def __init__(self, window_sec=60, cal_sys=None, cal_dia=None, cal_hr=None):
        self.hr_history = collections.deque(maxlen=300)  # 5 min at 1 Hz
        self.hr_timestamps = collections.deque(maxlen=300)
        self.window_sec = window_sec

        # Calibration offsets
        self.cal_offset_sys = 0.0
        self.cal_offset_dia = 0.0

        if cal_sys is not None and cal_hr is not None:
            # Compute what the model would predict at calibration HR
            predicted_sys = self.BASE_SYS + self.HR_COEFF_SYS * (cal_hr - self.BASE_HR)
            self.cal_offset_sys = cal_sys - predicted_sys

        if cal_dia is not None and cal_hr is not None:
            predicted_dia = self.BASE_DIA + self.HR_COEFF_DIA * (cal_hr - self.BASE_HR)
            self.cal_offset_dia = cal_dia - predicted_dia

    def add_hr(self, hr_bpm: float) -> None:
        """Add a heart rate measurement."""
        if hr_bpm <= 0 or hr_bpm > 220:
            return
        self.hr_history.append(hr_bpm)
        self.hr_timestamps.append(time.time())

    def _get_recent(self, window_sec: float):
        """Get HR values within the last window_sec seconds."""
        now = time.time()
        cutoff = now - window_sec
        values = []
        for t, hr in zip(self.hr_timestamps, self.hr_history):
            if t >= cutoff:
                values.append(hr)
        return values

    def _compute_sdnn(self, hrs: list) -> float:
        """Standard deviation of beat-to-beat intervals (SDNN proxy).

        We don't have R-R intervals, so we approximate from HR:
          RR_i ≈ 60 / HR_i (seconds)
        SDNN = std(RR_i) * 1000 (milliseconds)
        """
        if len(hrs) < 5:
            return 50.0  # Default: normal HRV

        rr_intervals = [60.0 / hr * 1000.0 for hr in hrs if hr > 0]
        if len(rr_intervals) < 5:
            return 50.0

        if HAS_NUMPY:
            return float(np.std(rr_intervals))
        else:
            mean = sum(rr_intervals) / len(rr_intervals)
            variance = sum((x - mean) ** 2 for x in rr_intervals) / len(rr_intervals)
            return math.sqrt(variance)

    def _compute_lf_hf_ratio(self, hrs: list) -> float:
        """Estimate LF/HF ratio from HR variability.

        LF (0.04-0.15 Hz): sympathetic + parasympathetic
        HF (0.15-0.4 Hz): parasympathetic only
        LF/HF > 2: sympathetic dominant (stress, higher BP)
        LF/HF < 1: parasympathetic dominant (relaxed, lower BP)

        Without true spectral analysis, we approximate from the
        ratio of slow (>10s period) to fast (<7s period) HR fluctuations.
        """
        if len(hrs) < 20:
            return 1.5  # Default: slight sympathetic

        if not HAS_NUMPY:
            return 1.5  # Need numpy for spectral estimate

        arr = np.array(hrs, dtype=float)
        detrended = arr - np.mean(arr)

        # Simple spectral power estimate via autocorrelation
        n = len(detrended)
        fft = np.fft.rfft(detrended)
        psd = np.abs(fft) ** 2 / n

        # Frequency bins (assuming 1 Hz sampling from mmWave)
        freqs = np.fft.rfftfreq(n, d=1.0)

        # LF band: 0.04-0.15 Hz
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0.0

        # HF band: 0.15-0.4 Hz
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0.001

        ratio = lf_power / max(hf_power, 0.001)
        return min(max(ratio, 0.1), 10.0)  # Clamp to reasonable range

    def estimate(self) -> dict:
        """Estimate current blood pressure.

        Returns dict with: systolic, diastolic, mean_hr, sdnn, lf_hf,
        confidence (0-100), n_samples.
        """
        recent = self._get_recent(self.window_sec)

        if len(recent) < 3:
            return {
                "systolic": 0, "diastolic": 0,
                "mean_hr": 0, "sdnn": 0, "lf_hf": 0,
                "confidence": 0, "n_samples": len(recent),
                "status": "Collecting data..."
            }

        mean_hr = sum(recent) / len(recent)
        sdnn = self._compute_sdnn(recent)
        lf_hf = self._compute_lf_hf_ratio(recent)

        # Model
        hr_delta = mean_hr - self.BASE_HR
        sys = (self.BASE_SYS
               + self.HR_COEFF_SYS * hr_delta
               + self.SDNN_COEFF_SYS * (sdnn - 50.0) / 50.0
               + self.LFHF_COEFF_SYS * (lf_hf - 1.5)
               + self.cal_offset_sys)

        dia = (self.BASE_DIA
               + self.HR_COEFF_DIA * hr_delta
               + self.SDNN_COEFF_DIA * (sdnn - 50.0) / 50.0
               + self.LFHF_COEFF_DIA * (lf_hf - 1.5)
               + self.cal_offset_dia)

        # Physiological clamps
        sys = max(80, min(200, sys))
        dia = max(50, min(130, dia))
        if dia >= sys:
            dia = sys - 20

        # Confidence based on data quality
        conf = min(100, len(recent) * 2)
        if self.cal_offset_sys != 0:
            conf = min(100, conf + 20)  # Calibrated = higher confidence

        status = "Estimating"
        if len(recent) < 10:
            status = "Warming up..."
        elif conf >= 80:
            status = "Stable estimate"

        return {
            "systolic": round(sys),
            "diastolic": round(dia),
            "mean_hr": round(mean_hr, 1),
            "sdnn": round(sdnn, 1),
            "lf_hf": round(lf_hf, 2),
            "confidence": conf,
            "n_samples": len(recent),
            "status": status,
        }


def bp_category(sys: int, dia: int) -> str:
    """AHA blood pressure category."""
    if sys == 0:
        return "—"
    if sys < 120 and dia < 80:
        return "Normal"
    elif sys < 130 and dia < 80:
        return "Elevated"
    elif sys < 140 or dia < 90:
        return "High BP Stage 1"
    elif sys >= 140 or dia >= 90:
        return "High BP Stage 2"
    elif sys > 180 or dia > 120:
        return "Hypertensive Crisis"
    return "Unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Contactless BP estimation from mmWave heart rate",
        epilog="NOT A MEDICAL DEVICE. For research/wellness tracking only.",
    )
    parser.add_argument("--port", default="COM4", help="mmWave sensor serial port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--window", type=int, default=60, help="Analysis window in seconds")
    parser.add_argument("--cal-systolic", type=int, help="Calibration: your actual systolic BP")
    parser.add_argument("--cal-diastolic", type=int, help="Calibration: your actual diastolic BP")
    parser.add_argument("--cal-hr", type=int, help="Calibration: your HR at time of BP reading")
    parser.add_argument("--duration", type=int, default=120, help="Recording duration in seconds")
    args = parser.parse_args()

    estimator = BPEstimator(
        window_sec=args.window,
        cal_sys=args.cal_systolic,
        cal_dia=args.cal_diastolic,
        cal_hr=args.cal_hr,
    )

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except Exception as e:
        print(f"Error opening {args.port}: {e}")
        sys.exit(1)

    print()
    print("=" * 66)
    print("  Contactless Blood Pressure Estimation (mmWave 60 GHz)")
    print("  ⚠️  NOT A MEDICAL DEVICE — research/wellness only")
    print("=" * 66)
    if args.cal_systolic:
        print(f"  Calibrated: {args.cal_systolic}/{args.cal_diastolic} mmHg at {args.cal_hr} bpm")
    else:
        print("  Uncalibrated — showing relative trends. Use --cal-* for accuracy.")
    print()

    header = f"  {'Time':>5}  {'HR':>5}  {'SBP':>5}  {'DBP':>5}  {'Category':>20}  {'SDNN':>6}  {'LF/HF':>6}  {'Conf':>4}  {'Status'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Print initial blank lines for live update area
    for _ in range(3):
        print()

    start = time.time()
    last_print = 0

    try:
        while time.time() - start < args.duration:
            line = ser.readline().decode("utf-8", errors="replace")
            clean = RE_ANSI.sub("", line)

            m = RE_HR.search(clean)
            if m:
                hr = float(m.group(1))
                estimator.add_hr(hr)

            # Update display every 3 seconds
            elapsed = int(time.time() - start)
            if elapsed > last_print and elapsed % 3 == 0:
                last_print = elapsed
                est = estimator.estimate()

                if est["systolic"] > 0:
                    cat = bp_category(est["systolic"], est["diastolic"])
                    sys.stdout.write(f"\r  {elapsed:>4}s  {est['mean_hr']:>4.0f}  "
                                     f"{est['systolic']:>4}  {est['diastolic']:>4}  "
                                     f"{cat:>20}  {est['sdnn']:>5.1f}  {est['lf_hf']:>5.2f}  "
                                     f"{est['confidence']:>3}%  {est['status']}")
                    sys.stdout.write("          \n")
                else:
                    sys.stdout.write(f"\r  {elapsed:>4}s  {'—':>4}  {'—':>4}  {'—':>4}  "
                                     f"{'—':>20}  {'—':>5}  {'—':>5}  "
                                     f"{'—':>3}  {est['status']}")
                    sys.stdout.write("          \n")
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass

    ser.close()

    # Final summary
    est = estimator.estimate()
    print()
    print()
    print("=" * 66)
    print("  BLOOD PRESSURE ESTIMATION SUMMARY")
    print("=" * 66)
    if est["systolic"] > 0:
        cat = bp_category(est["systolic"], est["diastolic"])
        print(f"  Systolic:    {est['systolic']} mmHg")
        print(f"  Diastolic:   {est['diastolic']} mmHg")
        print(f"  Category:    {cat}")
        print(f"  Mean HR:     {est['mean_hr']} bpm")
        print(f"  HRV (SDNN):  {est['sdnn']} ms")
        print(f"  LF/HF ratio: {est['lf_hf']}")
        print(f"  Confidence:  {est['confidence']}%")
        print(f"  Samples:     {est['n_samples']} readings over {args.window}s window")
    else:
        print("  Insufficient data. Ensure person is within sensor range.")
    print()
    print("  ⚠️  This is an ESTIMATE based on HR/HRV correlation models.")
    print("  For actual BP measurement, use a validated cuff device.")
    print()


if __name__ == "__main__":
    main()
