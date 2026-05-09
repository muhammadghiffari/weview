#!/usr/bin/env python3
"""
Real-Time Stress Monitor via Heart Rate Variability (HRV)

Reads heart rate from MR60BHA2 mmWave radar and computes HRV metrics
to estimate stress level continuously.

HRV Science:
  - SDNN < 50ms = high stress / low parasympathetic tone
  - SDNN 50-100ms = moderate
  - SDNN > 100ms = relaxed / high vagal tone
  - RMSSD: successive difference metric, more sensitive to acute stress

Usage:
    python examples/stress/hrv_stress_monitor.py --port COM4
"""

import argparse
import collections
import math
import re
import serial
import sys
import time

RE_HR = re.compile(r"'Real-time heart rate'.*?(\d+\.?\d*)\s*bpm", re.IGNORECASE)
RE_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def compute_hrv(hr_values):
    """Compute HRV metrics from HR time series."""
    if len(hr_values) < 5:
        return {"sdnn": 0, "rmssd": 0, "mean_hr": 0, "stress": "—"}

    rr = [60000.0 / h for h in hr_values if h > 0]
    if len(rr) < 5:
        return {"sdnn": 0, "rmssd": 0, "mean_hr": 0, "stress": "—"}

    mean_rr = sum(rr) / len(rr)
    sdnn = math.sqrt(sum((x - mean_rr) ** 2 for x in rr) / len(rr))

    # RMSSD: root mean square of successive differences
    diffs = [(rr[i+1] - rr[i]) ** 2 for i in range(len(rr) - 1)]
    rmssd = math.sqrt(sum(diffs) / len(diffs)) if diffs else 0

    mean_hr = sum(hr_values) / len(hr_values)

    if sdnn < 30:
        stress = "HIGH STRESS"
    elif sdnn < 50:
        stress = "Moderate Stress"
    elif sdnn < 80:
        stress = "Mild Stress"
    elif sdnn < 100:
        stress = "Relaxed"
    else:
        stress = "Very Relaxed"

    return {"sdnn": sdnn, "rmssd": rmssd, "mean_hr": mean_hr, "stress": stress}


def stress_bar(sdnn, width=30):
    """Visual stress bar: more filled = more stressed."""
    level = max(0, min(1, 1.0 - sdnn / 120.0))
    filled = int(level * width)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {level*100:.0f}%"


def main():
    parser = argparse.ArgumentParser(description="HRV Stress Monitor (mmWave)")
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--window", type=int, default=60, help="HRV window in seconds")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)

    print()
    print("=" * 60)
    print("  Real-Time Stress Monitor (mmWave HRV)")
    print("  Sit still within 1m. Lower stress = higher HRV.")
    print("=" * 60)
    print()

    hr_buffer = collections.deque(maxlen=args.window)
    start = time.time()
    last_print = 0
    min_stress = 999.0
    max_stress = 0.0
    readings = []

    try:
        while time.time() - start < args.duration:
            line = ser.readline().decode("utf-8", errors="replace")
            clean = RE_ANSI.sub("", line)

            m = RE_HR.search(clean)
            if m:
                hr = float(m.group(1))
                if 30 < hr < 200:
                    hr_buffer.append(hr)

            elapsed = int(time.time() - start)
            if elapsed > last_print and elapsed % 5 == 0 and len(hr_buffer) >= 3:
                last_print = elapsed
                hrv = compute_hrv(list(hr_buffer))
                bar = stress_bar(hrv["sdnn"])
                readings.append(hrv)

                if hrv["sdnn"] > 0:
                    min_stress = min(min_stress, hrv["sdnn"])
                    max_stress = max(max_stress, hrv["sdnn"])

                print(f"  {elapsed:>4}s | HR {hrv['mean_hr']:>4.0f} | "
                      f"SDNN {hrv['sdnn']:>5.1f}ms | RMSSD {hrv['rmssd']:>5.1f}ms | "
                      f"{hrv['stress']:<16} | {bar}")

    except KeyboardInterrupt:
        pass

    ser.close()

    print()
    print("=" * 60)
    print("  STRESS SESSION SUMMARY")
    print("=" * 60)
    if readings:
        avg_sdnn = sum(r["sdnn"] for r in readings) / len(readings)
        avg_rmssd = sum(r["rmssd"] for r in readings) / len(readings)
        avg_hr = sum(r["mean_hr"] for r in readings) / len(readings)
        final_stress = readings[-1]["stress"]

        print(f"  Duration:    {time.time()-start:.0f}s")
        print(f"  Avg HR:      {avg_hr:.0f} bpm")
        print(f"  Avg SDNN:    {avg_sdnn:.1f} ms {'(low — consider a break)' if avg_sdnn < 50 else '(healthy range)' if avg_sdnn > 70 else ''}")
        print(f"  Avg RMSSD:   {avg_rmssd:.1f} ms")
        print(f"  SDNN range:  {min_stress:.0f} - {max_stress:.0f} ms")
        print(f"  Assessment:  {final_stress}")
        print()
        print("  SDNN Guide: <30=high stress, 30-50=moderate, 50-100=normal, >100=relaxed")
    else:
        print("  No data collected. Ensure person is in range.")
    print()


if __name__ == "__main__":
    main()
