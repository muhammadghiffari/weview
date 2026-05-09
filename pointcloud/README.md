# WeView — Live 3D Point Cloud Viewer

Hosted at: https://ruvnet.github.io/WeView/pointcloud/

## Modes

- Default — synthetic in-browser demo (no backend, no network calls).
- `?backend=auto` — fetch from `/api/splats` on the same origin
  (only works when the viewer is served by `weview-pointcloud serve`).
- `?backend=<url>` — fetch from `<url>/api/splats`. The intended
  local-ESP32 use is `?backend=http://127.0.0.1:9880`: run
  `weview-pointcloud serve --bind 127.0.0.1:9880` on the same
  machine with your ESP32 streaming CSI to UDP port 3333, then
  visit the URL above. The local server's CorsLayer permits
  requests from `https://ruvnet.github.io`, and modern browsers
  permit HTTPS→127.0.0.1 mixed-content as a trustworthy origin.
  The "📡 Connect ESP32" button in the viewer prompts for this
  URL and persists it in localStorage.
- `?live=1` — require a live backend; show an offline message instead
  of falling back to the synthetic demo.

See ADR-094 for the deployment design.
