# Examples

Real-time sensing applications built on the WeView platform.

## Unified Dashboard (start here)

```bash
pip install pyserial numpy
python examples/weview_live.py --csi COM7 --mmwave COM4
```

The live dashboard auto-detects available sensors and displays fused vitals, environment data, and events in real-time. Works with any combination of sensors.

## Individual Examples

| Example | Sensors | What It Does |
|---------|---------|-------------|
| [**weview_live.py**](weview_live.py) | CSI + mmWave + Light | Unified dashboard: HR, BR, BP, stress, presence, light, RSSI |
| [Medical: Blood Pressure](medical/) | mmWave | Contactless BP estimation from HRV |
| [Medical: Vitals Suite](medical/vitals_suite.py) | mmWave | 10-in-1: HR, BR, BP, HRV, sleep stages, apnea, cough, snoring, activity, meditation |
| [Sleep: Apnea Screener](sleep/) | mmWave | Detects breathing cessation events, computes AHI |
| [Stress: HRV Monitor](stress/) | mmWave | Real-time stress level from heart rate variability |
| [Environment: Room Monitor](environment/) | CSI + mmWave | Occupancy, light, RF fingerprint, activity events |

## Hardware

| Port | Device | Cost | What It Provides |
|------|--------|------|-----------------|
| COM7 | ESP32-S3 (WiFi CSI) | ~$9 | Presence, motion, breathing, heart rate (through walls) |
| COM4 | ESP32-C6 + Seeed MR60BHA2 | ~$15 | Precise HR/BR, presence, distance, ambient light |

Either sensor works alone. Both together enable fusion (mmWave 80% + CSI 20%).

## Quick Start

```bash
pip install pyserial numpy

# Unified dashboard (recommended)
python examples/weview_live.py --csi COM7 --mmwave COM4

# Blood pressure estimation
python examples/medical/bp_estimator.py --port COM4

# Sleep apnea screening (run overnight)
python examples/sleep/apnea_screener.py --port COM4 --duration 28800

# Stress monitoring (workday session)
python examples/stress/hrv_stress_monitor.py --port COM4 --duration 3600

# Room environment monitor
python examples/environment/room_monitor.py --csi-port COM7 --mmwave-port COM4

# CSI only (no mmWave)
python examples/weview_live.py --csi COM7 --mmwave none
```
