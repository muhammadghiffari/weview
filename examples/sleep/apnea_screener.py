#!/usr/bin/env python3
"""
Sleep Apnea Screener — Contactless via 60 GHz mmWave

Monitors breathing rate from MR60BHA2 and detects apnea events
(breathing cessation > 10 seconds). Clinical threshold: > 5 events/hour
= Obstructive Sleep Apnea (mild), > 15 = moderate, > 30 = severe.

Usage:
    python examples/sleep/apnea_screener.py --port COM4
    python examples/sleep/apnea_screener.py --port COM4 --duration 3600  # 1 hour
"""

import argparse
import collections
import re
import serial
import sys
import time

RE_BR = re.compile(r"'Real-time respiratory rate'.*?(\d+\.?\d*)", re.IGNORECASE)
RE_HR = re.compile(r"'Real-time heart rate'.*?(\d+\.?\d*)", re.IGNORECASE)
RE_PRES = re.compile(r"'Person Information'.*?state\s+(ON|OFF)", re.IGNORECASE)
RE_ANSI = re.compile(r"\x1b\[[0-9;]*m")

APNEA_THRESHOLD_SEC = 10  # Breathing absent for >10s = apnea event
HYPOPNEA_BR = 6.0         # BR < 6/min = hypopnea (shallow breathing)


def main():
    parser = argparse.ArgumentParser(description="Sleep Apnea Screener (mmWave)")
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)

    print()
    print("=" * 60)
    print("  Sleep Apnea Screener (60 GHz mmWave)")
    print("  Lie still within 1m of sensor. Monitoring breathing.")
    print("=" * 60)
    print()

    br_history = collections.deque(maxlen=600)
    apnea_events = []
    hypopnea_events = []
    last_br_time = time.time()
    last_br_value = 0.0
    last_hr = 0.0
    in_apnea = False
    apnea_start = 0.0
    start = time.time()
    last_print = 0

    try:
        while time.time() - start < args.duration:
            line = ser.readline().decode("utf-8", errors="replace")
            clean = RE_ANSI.sub("", line)

            m = RE_BR.search(clean)
            if m:
                br = float(m.group(1))
                br_history.append((time.time(), br))

                if br > 0:
                    last_br_time = time.time()
                    last_br_value = br

                    if in_apnea:
                        duration = time.time() - apnea_start
                        apnea_events.append(duration)
                        print(f"  ** APNEA EVENT ENDED: {duration:.1f}s **")
                        in_apnea = False

                    if br < HYPOPNEA_BR and br > 0:
                        hypopnea_events.append(br)

                elif br == 0 and not in_apnea:
                    gap = time.time() - last_br_time
                    if gap >= APNEA_THRESHOLD_SEC:
                        in_apnea = True
                        apnea_start = last_br_time
                        print(f"  ** APNEA DETECTED at {int(time.time()-start)}s (no breath for {gap:.0f}s) **")

            m = RE_HR.search(clean)
            if m:
                last_hr = float(m.group(1))

            elapsed = int(time.time() - start)
            if elapsed > last_print and elapsed % 10 == 0:
                last_print = elapsed
                gap = time.time() - last_br_time
                status = "APNEA" if in_apnea else ("OK" if gap < 5 else f"gap {gap:.0f}s")
                print(f"  {elapsed:>4}s | BR {last_br_value:>4.0f}/min | HR {last_hr:>4.0f} | "
                      f"Apneas: {len(apnea_events)} | Hypopneas: {len(hypopnea_events)} | {status}")

    except KeyboardInterrupt:
        pass

    ser.close()
    duration_hr = (time.time() - start) / 3600.0

    print()
    print("=" * 60)
    print("  APNEA SCREENING RESULTS")
    print("=" * 60)
    ahi = (len(apnea_events) + len(hypopnea_events)) / max(duration_hr, 0.01)
    print(f"  Duration:      {time.time()-start:.0f}s ({duration_hr*60:.1f} min)")
    print(f"  Apnea events:  {len(apnea_events)} (breathing absent > {APNEA_THRESHOLD_SEC}s)")
    print(f"  Hypopneas:     {len(hypopnea_events)} (BR < {HYPOPNEA_BR}/min)")
    print(f"  AHI estimate:  {ahi:.1f} events/hour")
    print()
    if ahi < 5:
        print("  Classification: Normal (AHI < 5)")
    elif ahi < 15:
        print("  Classification: Mild OSA (AHI 5-14)")
    elif ahi < 30:
        print("  Classification: Moderate OSA (AHI 15-29)")
    else:
        print("  Classification: Severe OSA (AHI >= 30)")
    print()
    print("  NOT A MEDICAL DEVICE. Consult a sleep specialist for diagnosis.")
    print()


if __name__ == "__main__":
    main()
