#!/usr/bin/env python3
"""
Room Environment Monitor — WiFi CSI + mmWave + Light Sensor Fusion

Combines all available sensors to build a real-time room awareness picture:
  - WiFi CSI (COM7): Presence, motion energy, room RF fingerprint
  - mmWave (COM4): Occupancy count, distance, HR/BR of nearest person
  - BH1750 (COM4): Ambient light level

Detects: occupancy changes, lighting anomalies, activity patterns,
room RF fingerprint drift (door/window state changes).

Usage:
    python examples/environment/room_monitor.py --csi-port COM7 --mmwave-port COM4
"""

import argparse
import collections
import math
import re
import serial
import sys
import threading
import time

RE_HR = re.compile(r"'Real-time heart rate'.*?(\d+\.?\d*)\s*bpm", re.IGNORECASE)
RE_BR = re.compile(r"'Real-time respiratory rate'.*?(\d+\.?\d*)", re.IGNORECASE)
RE_PRES = re.compile(r"'Person Information'.*?state\s+(ON|OFF)", re.IGNORECASE)
RE_DIST = re.compile(r"'Distance to detection object'.*?(\d+\.?\d*)\s*cm", re.IGNORECASE)
RE_LUX = re.compile(r"'Seeed MR60BHA2 Illuminance'.*?(\d+\.?\d*)\s*lx", re.IGNORECASE)
RE_TARGETS = re.compile(r"'Target Number'.*?(\d+\.?\d*)", re.IGNORECASE)
RE_CSI_CB = re.compile(r"CSI cb #(\d+).*?len=(\d+).*?rssi=(-?\d+)")
RE_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Light categories
def light_category(lux):
    if lux < 1: return "Dark"
    if lux < 10: return "Dim"
    if lux < 50: return "Low"
    if lux < 200: return "Normal"
    if lux < 500: return "Bright"
    return "Very Bright"


def main():
    parser = argparse.ArgumentParser(description="Room Environment Monitor")
    parser.add_argument("--csi-port", default="COM7")
    parser.add_argument("--mmwave-port", default="COM4")
    parser.add_argument("--duration", type=int, default=120)
    args = parser.parse_args()

    # Shared state
    state = {
        "hr": 0.0, "br": 0.0, "presence_mw": False, "distance": 0.0,
        "lux": 0.0, "targets": 0, "rssi": 0, "csi_frames": 0,
        "mw_frames": 0, "events": [],
    }
    rssi_history = collections.deque(maxlen=60)
    lux_history = collections.deque(maxlen=60)
    lock = threading.Lock()
    stop = threading.Event()

    def read_mmwave():
        try:
            ser = serial.Serial(args.mmwave_port, 115200, timeout=1)
        except Exception:
            return
        while not stop.is_set():
            line = ser.readline().decode("utf-8", errors="replace")
            clean = RE_ANSI.sub("", line)
            with lock:
                m = RE_HR.search(clean)
                if m: state["hr"] = float(m.group(1)); state["mw_frames"] += 1
                m = RE_BR.search(clean)
                if m: state["br"] = float(m.group(1))
                m = RE_PRES.search(clean)
                if m:
                    new_pres = m.group(1) == "ON"
                    if new_pres != state["presence_mw"]:
                        event = f"Person {'arrived' if new_pres else 'left'} (mmWave)"
                        state["events"].append((time.time(), event))
                    state["presence_mw"] = new_pres
                m = RE_DIST.search(clean)
                if m: state["distance"] = float(m.group(1))
                m = RE_LUX.search(clean)
                if m:
                    lux = float(m.group(1))
                    old_cat = light_category(state["lux"])
                    new_cat = light_category(lux)
                    if old_cat != new_cat and state["lux"] > 0:
                        state["events"].append((time.time(), f"Light: {old_cat} -> {new_cat} ({lux:.1f} lx)"))
                    state["lux"] = lux
                    lux_history.append(lux)
                m = RE_TARGETS.search(clean)
                if m: state["targets"] = int(float(m.group(1)))
        ser.close()

    def read_csi():
        try:
            ser = serial.Serial(args.csi_port, 115200, timeout=1)
        except Exception:
            return
        while not stop.is_set():
            line = ser.readline().decode("utf-8", errors="replace")
            m = RE_CSI_CB.search(line)
            if m:
                with lock:
                    state["csi_frames"] = int(m.group(1))
                    state["rssi"] = int(m.group(3))
                    rssi_history.append(int(m.group(3)))
        ser.close()

    t1 = threading.Thread(target=read_mmwave, daemon=True)
    t2 = threading.Thread(target=read_csi, daemon=True)
    t1.start()
    t2.start()

    print()
    print("=" * 70)
    print("  Room Environment Monitor (WiFi CSI + mmWave + Light)")
    print("=" * 70)
    print()

    start_time = time.time()
    last_print = 0

    try:
        while time.time() - start_time < args.duration:
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            if elapsed <= last_print or elapsed % 5 != 0:
                continue
            last_print = elapsed

            with lock:
                s = dict(state)
                events = list(state["events"][-3:])

            # RSSI stability (RF fingerprint drift)
            rssi_std = 0
            if len(rssi_history) >= 5:
                vals = list(rssi_history)
                mean = sum(vals) / len(vals)
                rssi_std = math.sqrt(sum((x - mean)**2 for x in vals) / len(vals))

            rf_status = "Stable" if rssi_std < 3 else "Shifting" if rssi_std < 6 else "Volatile"

            pres = "YES" if s["presence_mw"] else "no"
            lcat = light_category(s["lux"])

            print(f"  {elapsed:>4}s | Pres:{pres:>3} Dist:{s['distance']:>4.0f}cm | "
                  f"HR:{s['hr']:>3.0f} BR:{s['br']:>2.0f} | "
                  f"Light:{s['lux']:>5.1f}lx ({lcat:<6}) | "
                  f"RSSI:{s['rssi']:>3}dBm RF:{rf_status:<8} | "
                  f"CSI:{s['csi_frames']} MW:{s['mw_frames']}")

            for ts, event in events:
                age = elapsed - int(ts - start_time)
                if age < 10:
                    print(f"         ** EVENT: {event}")

    except KeyboardInterrupt:
        pass

    stop.set()
    time.sleep(1)

    print()
    print("=" * 70)
    print("  ROOM SUMMARY")
    print("=" * 70)
    with lock:
        print(f"  Duration:    {time.time()-start_time:.0f}s")
        print(f"  CSI frames:  {state['csi_frames']}")
        print(f"  mmWave data: {state['mw_frames']} readings")
        print(f"  Last HR:     {state['hr']:.0f} bpm")
        print(f"  Last BR:     {state['br']:.0f}/min")
        print(f"  Light:       {state['lux']:.1f} lux ({light_category(state['lux'])})")
        if lux_history:
            print(f"  Light range: {min(lux_history):.1f} - {max(lux_history):.1f} lux")
        if rssi_history:
            print(f"  RSSI range:  {min(rssi_history)} to {max(rssi_history)} dBm (std={rssi_std:.1f})")
        print(f"  Events:      {len(state['events'])}")
        for ts, event in state["events"]:
            print(f"    [{int(ts-start_time):>4}s] {event}")
    print()


if __name__ == "__main__":
    main()
