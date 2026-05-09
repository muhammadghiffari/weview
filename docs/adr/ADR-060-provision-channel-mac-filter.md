# ADR-060: Provision Channel Override and MAC Address Filtering

- **Status:** Accepted
- **Date:** 2026-03-12
- **Issues:** [#247](https://github.com/ruvnet/WeView/issues/247), [#229](https://github.com/ruvnet/WeView/issues/229)

## Context

Two related provisioning gaps were reported by users:

1. **Channel mismatch (Issue #247):** The CSI collector initializes on the
   Kconfig default channel (typically 6), even when the ESP32 connects to an AP
   on a different channel (e.g. 11). On managed networks where the user cannot
   change the router channel, this makes nodes undiscoverable. The
   `provision.py` script has no `--channel` argument.

2. **Missing MAC filter (Issue #229):** The v0.2.0 release notes documented a
   `--filter-mac` argument for `provision.py`, but it was never implemented.
   The firmware's CSI callback accepts frames from all sources, causing signal
   mixing in multi-AP environments.

## Decision

### Channel configuration

- Add `--channel` argument to `provision.py` that writes a `csi_channel` key
  (u8) to NVS.
- In `nvs_config.c`, read the `csi_channel` key and override
  `channel_list[0]` when present.
- In `csi_collector_init()`, after WiFi connects, auto-detect the AP channel
  via `esp_wifi_sta_get_ap_info()` and use it as the default CSI channel when
  no NVS override is set. This ensures the CSI collector always matches the
  connected AP's channel without requiring manual provisioning.

### MAC address filtering

- Add `--filter-mac` argument to `provision.py` that writes a `filter_mac`
  key (6-byte blob) to NVS.
- In `nvs_config.h`, add a `filter_mac[6]` field and `filter_mac_set` flag.
- In `nvs_config.c`, read the `filter_mac` blob from NVS.
- In the CSI callback (`wifi_csi_callback`), if `filter_mac_set` is true,
  compare the source MAC from the received frame against the configured MAC
  and drop non-matching frames.

### Provisioning flow

```
python provision.py --port COM7 --channel 11
python provision.py --port COM7 --filter-mac "AA:BB:CC:DD:EE:FF"
python provision.py --port COM7 --channel 11 --filter-mac "AA:BB:CC:DD:EE:FF"
```

## Consequences

- Users on managed networks can force the CSI channel to match their AP
- Multi-AP environments can filter CSI to a single source
- Auto-channel detection eliminates the most common misconfiguration
- Backward compatible: existing provisioned nodes without these keys behave
  as before (use Kconfig default channel, accept all MACs)
