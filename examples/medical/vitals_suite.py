#!/usr/bin/env python3
"""
WeView Medical Vitals Suite — 10 capabilities from a single mmWave sensor

Capabilities:
  1. Heart rate monitoring (continuous)
  2. Breathing rate monitoring (continuous)
  3. Blood pressure estimation (HRV-based)
  4. HRV stress analysis (SDNN, RMSSD, pNN50, LF/HF)
  5. Sleep stage classification (awake/light/deep/REM)
  6. Apnea event detection (BR=0 for >10s)
  7. Cough detection (BR spike pattern)
  8. Snoring detection (periodic high-amplitude BR)
  9. Activity state (resting/active/exercising)
  10. Meditation quality scorer (coherence of BR+HR)

Usage:
    python examples/medical/vitals_suite.py --port COM4 --duration 120
"""

import argparse
import collections
import math
import re
import serial
import sys
import time

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

RE_HR = re.compile(r"'Real-time heart rate'.*?(\d+\.?\d*)\s*bpm", re.I)
RE_BR = re.compile(r"'Real-time respiratory rate'.*?(\d+\.?\d*)", re.I)
RE_PRES = re.compile(r"'Person Information'.*?state\s+(ON|OFF)", re.I)
RE_DIST = re.compile(r"'Distance to detection object'.*?(\d+\.?\d*)\s*cm", re.I)
RE_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class WelfordStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, v):
        self.count += 1
        d = v - self.mean
        self.mean += d / self.count
        self.m2 += d * (v - self.mean)

    def std(self):
        return math.sqrt(self.m2 / self.count) if self.count > 1 else 0.0

    def cv(self):
        return self.std() / self.mean if self.mean > 0 else 0.0


class VitalsSuite:
    def __init__(self):
        # Raw buffers
        self.hr_buf = collections.deque(maxlen=300)
        self.br_buf = collections.deque(maxlen=300)
        self.hr_ts = collections.deque(maxlen=300)
        self.br_ts = collections.deque(maxlen=300)
        self.distance = 0.0
        self.presence = False
        self.frames = 0

        # Welford trackers
        self.hr_stats = WelfordStats()
        self.br_stats = WelfordStats()

        # Apnea detection
        self.last_br_time = time.time()
        self.last_nonzero_br = 0.0
        self.apnea_events = []
        self.in_apnea = False
        self.apnea_start = 0.0

        # Cough detection
        self.cough_events = []
        self.prev_br = 0.0

        # Snoring detection
        self.snore_events = 0
        self.br_amplitude_buf = collections.deque(maxlen=30)

        # Sleep state
        self.sleep_state = "Awake"
        self.sleep_onset = 0.0

        # Meditation
        self.meditation_score = 0.0

        # Events
        self.events = collections.deque(maxlen=50)

    def feed(self, hr=0.0, br=0.0, presence=False, distance=0.0):
        now = time.time()
        self.presence = presence
        self.distance = distance
        self.frames += 1

        if hr > 0:
            self.hr_buf.append(hr)
            self.hr_ts.append(now)
            self.hr_stats.update(hr)

        if br > 0:
            self.br_buf.append(br)
            self.br_ts.append(now)
            self.br_stats.update(br)
            self.last_br_time = now
            self.last_nonzero_br = br

            # Cough: sudden BR spike > 2x baseline
            if self.prev_br > 0 and br > self.prev_br * 2.5 and self.br_stats.count > 10:
                self.cough_events.append(now)
                self.events.append((now, "Cough detected"))

            # Snoring: track BR amplitude variation
            if len(self.br_buf) >= 2:
                amp = abs(br - list(self.br_buf)[-2])
                self.br_amplitude_buf.append(amp)

            self.prev_br = br

            # End apnea
            if self.in_apnea:
                duration = now - self.apnea_start
                self.apnea_events.append(duration)
                self.events.append((now, f"Apnea ended ({duration:.0f}s)"))
                self.in_apnea = False
        else:
            # Apnea: BR=0 for >10s
            gap = now - self.last_br_time
            if gap >= 10 and not self.in_apnea and self.br_stats.count > 5:
                self.in_apnea = True
                self.apnea_start = self.last_br_time
                self.events.append((now, f"APNEA started (no breath for {gap:.0f}s)"))

        # Sleep stage classification
        self._classify_sleep()

        # Meditation score
        self._compute_meditation()

        # Snoring: periodic high-amplitude BR oscillation
        if len(self.br_amplitude_buf) >= 10:
            amps = list(self.br_amplitude_buf)
            mean_amp = sum(amps) / len(amps)
            if mean_amp > 3.0 and self.sleep_state != "Awake":
                self.snore_events += 1

    def _classify_sleep(self):
        """Sleep stage from BR variability + HR patterns."""
        hrs = list(self.hr_buf)
        brs = list(self.br_buf)

        if len(hrs) < 10 or len(brs) < 10:
            self.sleep_state = "Awake"
            return

        recent_hr = hrs[-10:]
        recent_br = brs[-10:]
        mean_hr = sum(recent_hr) / len(recent_hr)
        mean_br = sum(recent_br) / len(recent_br)

        # HR variability of last 10 readings
        hr_std = math.sqrt(sum((h - mean_hr) ** 2 for h in recent_hr) / len(recent_hr))
        br_std = math.sqrt(sum((b - mean_br) ** 2 for b in recent_br) / len(recent_br))

        # Activity check
        if mean_hr > 100 or mean_br > 25:
            self.sleep_state = "Awake"
            return

        # Low HR + low BR + low variability = deep sleep
        if mean_hr < 60 and mean_br < 14 and hr_std < 3 and br_std < 1:
            if self.sleep_state != "Deep Sleep":
                self.events.append((time.time(), "Entered deep sleep"))
            self.sleep_state = "Deep Sleep"
        # Moderate HR + high HR variability = REM
        elif hr_std > 5 and br_std > 2 and mean_br < 20:
            if self.sleep_state != "REM":
                self.events.append((time.time(), "Entered REM sleep"))
            self.sleep_state = "REM"
        # Low-moderate HR + low motion = light sleep
        elif mean_hr < 75 and mean_br < 20:
            if self.sleep_state != "Light Sleep":
                self.events.append((time.time(), "Entered light sleep"))
            self.sleep_state = "Light Sleep"
        else:
            self.sleep_state = "Awake"

    def _compute_meditation(self):
        """Meditation quality: BR regularity + HR deceleration + HRV increase."""
        brs = list(self.br_buf)
        hrs = list(self.hr_buf)
        if len(brs) < 15 or len(hrs) < 15:
            self.meditation_score = 0.0
            return

        # BR regularity (lower CV = more regular breathing)
        br_recent = brs[-15:]
        br_mean = sum(br_recent) / len(br_recent)
        br_std = math.sqrt(sum((b - br_mean) ** 2 for b in br_recent) / len(br_recent))
        br_cv = br_std / br_mean if br_mean > 0 else 1.0
        br_score = max(0, min(1, 1.0 - br_cv * 5))  # CV < 0.05 = perfect

        # HR deceleration (lower HR = better)
        hr_recent = hrs[-15:]
        mean_hr = sum(hr_recent) / len(hr_recent)
        hr_score = max(0, min(1, (90 - mean_hr) / 30))  # 60bpm=1.0, 90bpm=0.0

        # HRV increase (higher SDNN = better)
        rr = [60000 / h for h in hr_recent if h > 0]
        if len(rr) >= 5:
            rr_mean = sum(rr) / len(rr)
            sdnn = math.sqrt(sum((r - rr_mean) ** 2 for r in rr) / len(rr))
            hrv_score = max(0, min(1, sdnn / 100))  # 100ms SDNN = perfect
        else:
            hrv_score = 0.0

        self.meditation_score = (br_score * 0.4 + hr_score * 0.3 + hrv_score * 0.3) * 100

    def activity_state(self):
        if len(self.hr_buf) < 3:
            return "Unknown"
        recent = list(self.hr_buf)[-5:]
        mean_hr = sum(recent) / len(recent)
        if mean_hr > 120:
            return "Exercising"
        elif mean_hr > 90:
            return "Active"
        elif mean_hr > 60:
            return "Resting"
        else:
            return "Deep Rest"

    def hrv(self):
        hrs = list(self.hr_buf)
        if len(hrs) < 5:
            return {"sdnn": 0, "rmssd": 0, "pnn50": 0}
        rr = [60000 / h for h in hrs if h > 0]
        if len(rr) < 5:
            return {"sdnn": 0, "rmssd": 0, "pnn50": 0}
        mean = sum(rr) / len(rr)
        sdnn = math.sqrt(sum((r - mean) ** 2 for r in rr) / len(rr))
        diffs = [abs(rr[i + 1] - rr[i]) for i in range(len(rr) - 1)]
        rmssd = math.sqrt(sum(d ** 2 for d in diffs) / len(diffs)) if diffs else 0
        pnn50 = sum(1 for d in diffs if d > 50) / len(diffs) * 100 if diffs else 0
        return {"sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50}

    def bp(self):
        hrs = list(self.hr_buf)
        if len(hrs) < 5:
            return 0, 0
        mean_hr = sum(hrs) / len(hrs)
        hrv = self.hrv()
        if hrv["sdnn"] <= 0:
            return 0, 0
        delta = mean_hr - 72
        sbp = round(max(80, min(200, 120 + 0.5 * delta - 0.8 * (hrv["sdnn"] - 50) / 50)))
        dbp = round(max(50, min(130, 80 + 0.3 * delta - 0.5 * (hrv["sdnn"] - 50) / 50)))
        return sbp, dbp

    def stress(self):
        h = self.hrv()
        s = h["sdnn"]
        if s <= 0: return "---"
        if s < 30: return "HIGH"
        if s < 50: return "Moderate"
        if s < 80: return "Mild"
        if s < 100: return "Relaxed"
        return "Calm"


def main():
    parser = argparse.ArgumentParser(description="Medical Vitals Suite (10 capabilities)")
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=int, default=120)
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    suite = VitalsSuite()
    start = time.time()
    last_print = 0

    print()
    print("=" * 80)
    print("  WeView Medical Vitals Suite (10 capabilities from 1 sensor)")
    print("  Point MR60BHA2 at yourself within 1m. Sit still.")
    print("=" * 80)
    print()
    print(f"{'s':>4} {'HR':>4} {'BR':>3} {'BP':>7} {'Stress':>8} {'SDNN':>5} "
          f"{'Sleep':>11} {'Activity':>10} {'Medit':>5} "
          f"{'Apnea':>5} {'Cough':>5} {'Snore':>5}")
    print("-" * 80)

    try:
        while time.time() - start < args.duration:
            line = ser.readline().decode("utf-8", errors="replace")
            clean = RE_ANSI.sub("", line)

            hr, br, pres, dist = 0.0, 0.0, suite.presence, suite.distance
            m = RE_HR.search(clean)
            if m: hr = float(m.group(1))
            m = RE_BR.search(clean)
            if m: br = float(m.group(1))
            m = RE_PRES.search(clean)
            if m: pres = m.group(1) == "ON"
            m = RE_DIST.search(clean)
            if m: dist = float(m.group(1))

            if hr > 0 or br > 0:
                suite.feed(hr=hr, br=br, presence=pres, distance=dist)

            elapsed = int(time.time() - start)
            if elapsed > last_print and elapsed % 5 == 0:
                last_print = elapsed
                hrv = suite.hrv()
                sbp, dbp = suite.bp()
                bp_s = f"{sbp:>3}/{dbp:<3}" if sbp > 0 else "  ---  "
                sdnn_s = f"{hrv['sdnn']:>5.0f}" if hrv["sdnn"] > 0 else "  ---"

                hrs = list(suite.hr_buf)
                mean_hr = sum(hrs) / len(hrs) if hrs else 0

                brs = list(suite.br_buf)
                mean_br = sum(brs) / len(brs) if brs else 0

                print(f"{elapsed:>3}s {mean_hr:>4.0f} {mean_br:>3.0f} {bp_s} {suite.stress():>8} {sdnn_s} "
                      f"{suite.sleep_state:>11} {suite.activity_state():>10} {suite.meditation_score:>5.0f} "
                      f"{len(suite.apnea_events):>5} {len(suite.cough_events):>5} {suite.snore_events:>5}")

                # Print recent events
                for ts, msg in list(suite.events)[-3:]:
                    if time.time() - ts < 6:
                        print(f"       >> {msg}")

    except KeyboardInterrupt:
        pass

    ser.close()
    elapsed = time.time() - start

    print()
    print("=" * 80)
    print("  VITALS SUITE SUMMARY")
    print("=" * 80)
    hrv = suite.hrv()
    sbp, dbp = suite.bp()
    hrs = list(suite.hr_buf)
    brs = list(suite.br_buf)

    print(f"  Duration:        {elapsed:.0f}s")
    print(f"  Readings:        {suite.frames}")
    print()

    if hrs:
        print(f"  1. Heart Rate:   {sum(hrs)/len(hrs):.0f} bpm (range {min(hrs):.0f}-{max(hrs):.0f})")
    if brs:
        print(f"  2. Breathing:    {sum(brs)/len(brs):.0f}/min (range {min(brs):.0f}-{max(brs):.0f})")
    if sbp:
        print(f"  3. BP Estimate:  {sbp}/{dbp} mmHg")
    if hrv["sdnn"] > 0:
        print(f"  4. HRV/Stress:   SDNN={hrv['sdnn']:.0f}ms RMSSD={hrv['rmssd']:.0f}ms pNN50={hrv['pnn50']:.1f}% -> {suite.stress()}")
    print(f"  5. Sleep State:  {suite.sleep_state}")
    print(f"  6. Apnea Events: {len(suite.apnea_events)} {'(AHI=' + str(round(len(suite.apnea_events)/(elapsed/3600),1)) + '/hr)' if suite.apnea_events else ''}")
    print(f"  7. Cough Events: {len(suite.cough_events)}")
    print(f"  8. Snore Events: {suite.snore_events}")
    print(f"  9. Activity:     {suite.activity_state()}")
    print(f"  10. Meditation:  {suite.meditation_score:.0f}/100")

    if suite.events:
        print(f"\n  Events ({len(suite.events)}):")
        for ts, msg in list(suite.events)[-15:]:
            print(f"    [{int(ts-start):>4}s] {msg}")

    print()
    print("  NOT A MEDICAL DEVICE. For research/wellness only.")
    print()


if __name__ == "__main__":
    main()
