# 📡 We View — ESP32 Node Quick Reference
# Last updated: 2026-05-08 | All 3 nodes flashed & provisioned ✅

## Hardware Registry
# Node 1: COM5 | MAC e0:72:a1:d6:db:bc | TDM 0/3
# Node 2: COM3 | MAC dc:b4:d9:06:fa:34 | TDM 1/3
# Node 3: COM6 | MAC 14:c1:9f:28:b2:18 | TDM 2/3

## ============================================================
## GANTI WIFI? Edit 4 baris ini lalu jalankan seluruh script
## ============================================================

$SSID = "G.439 Dormitory"         # ← WiFi SSID
$PASS = "@G439.25"                # ← WiFi Password
$IP   = "10.9.8.178"              # ← PC IP (cek: ipconfig | sls IPv4)
$CH   = 153                       # ← Channel (cek: netsh wlan show interfaces | sls Channel)

## ============================================================
## JANGAN EDIT DI BAWAH INI
## ============================================================

cd "\\wsl.localhost\Ubuntu-22.04\home\kezman\WeView\firmware\esp32-csi-node"

# Node 1 — COM5
python provision.py `
  --port COM5 --ssid $SSID --password $PASS `
  --target-ip $IP --target-port 5005 `
  --node-id 1 --channel $CH `
  --tdm-slot 0 --tdm-total 3 --edge-tier 2

# Node 2 — COM3
python provision.py `
  --port COM3 --ssid $SSID --password $PASS `
  --target-ip $IP --target-port 5005 `
  --node-id 2 --channel $CH `
  --tdm-slot 1 --tdm-total 3 --edge-tier 2

# Node 3 — COM6
python provision.py `
  --port COM6 --ssid $SSID --password $PASS `
  --target-ip $IP --target-port 5005 `
  --node-id 3 --channel $CH `
  --tdm-slot 2 --tdm-total 3 --edge-tier 2

Write-Host "`n✅ All 3 nodes provisioned for: $SSID → $IP`:5005 (ch $CH)" -ForegroundColor Green
Write-Host "Cabut USB, colok power bank, jalankan server." -ForegroundColor Cyan
