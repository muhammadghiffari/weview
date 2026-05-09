# We View — ESP32 3-Node Setup & Operations Guide

> **3× ESP32-S3N16R8** | All nodes flashed & provisioned ✅  
> Last updated: 2026-05-08

---

## Node Registry

| Node | COM | MAC Address | Node ID | TDM Slot |
|------|-----|-------------|---------|----------|
| 🔴 1 | COM5 | `e0:72:a1:d6:db:bc` | 1 | 0/3 |
| 🔵 2 | COM3 | `dc:b4:d9:06:fa:34` | 2 | 1/3 |
| 🟢 3 | COM6 | `14:c1:9f:28:b2:18` | 3 | 2/3 |

All chips: ESP32-S3 (QFN56) rev v0.2, PSRAM 8MB, Flash 16MB

---

## 🔄 Ganti WiFi (Production / Pindah Venue)

**Ini yang paling sering dilakukan.** Firmware sudah di-flash, Anda hanya perlu update config WiFi.

### Langkah:
1. **Hubungkan laptop ke WiFi baru**
2. **Cek 2 info ini:**
   ```powershell
   ipconfig | Select-String "IPv4"              # → catat IP
   netsh wlan show interfaces | Select-String "Channel"  # → catat Channel
   ```
3. **Sambungkan 3 ESP32 ke laptop via USB**
4. **Edit & jalankan script:**
   ```powershell
   cd "\\wsl.localhost\Ubuntu-22.04\home\kezman\WeView\firmware\esp32-csi-node"
   
   # Edit file reprovision-all.ps1 — ganti 4 baris teratas:
   #   $SSID = "NamaWifiBaru"
   #   $PASS = "PasswordBaru"
   #   $IP   = "192.168.x.x"
   #   $CH   = 6
   
   .\reprovision-all.ps1
   ```
5. **Cabut USB, colok ke power bank, jalankan server**

> ⏱️ Total waktu re-provision: **~30 detik** untuk 3 node.

---

## 🚀 Jalankan Sistem

### Server (pilih salah satu)

**Option A — Native Rust (WSL):**
```bash
cd ~/WeView/v2
cargo run -p wifi-densepose-sensing-server -- --source esp32
# Server: HTTP localhost:8080 | WS localhost:8765 | UDP :5005
```

**Option B — Docker (PowerShell):**
```powershell
docker run -d --name weview-server `
  -p 3000:3000 -p 3001:3001 -p 5005:5005/udp `
  ruvnet/wifi-densepose:latest `
  --source esp32 --udp-port 5005 --http-port 3000 --ws-port 3001
```

### Dashboard
```bash
# Jika pakai Native Rust, buka terminal WSL kedua:
cd ~/WeView/dashboard && npm run dev
# Buka http://localhost:5173
```

### ESP32 Nodes
Colokkan ke power (USB power bank 5V/1A cukup). Akan otomatis connect WiFi dan streaming data.

---

## 📐 Penempatan Node

```
        [Node 1]           Bentuk segitiga
         /    \            di dalam ruangan
        /      \           
       /  AREA  \          Jarak ideal: 2-5m
      /  DETEKSI \         Tinggi: ~1.5m
     /            \        
[Node 2] ——————— [Node 3]
```

---

## 🔍 Verifikasi & Debug

### Cek serial output ESP32
```powershell
pip install pyserial    # sekali saja
python -m serial.tools.miniterm COM5 115200   # Node 1
python -m serial.tools.miniterm COM3 115200   # Node 2
python -m serial.tools.miniterm COM6 115200   # Node 3
```

### Cek server log
```bash
# Native Rust — lihat terminal langsung
# Docker:
docker logs weview-server -f
```

Sukses jika muncul:
```
ESP32 frame received from node_id=1
ESP32 frame received from node_id=2
ESP32 frame received from node_id=3
```

---

## 🛠️ Flash Ulang Firmware (Jarang Dilakukan)

Hanya perlu jika ada update firmware. Jalankan per node:

```powershell
cd "\\wsl.localhost\Ubuntu-22.04\home\kezman\WeView\firmware\esp32-csi-node\release_bins"

python -m esptool --chip esp32s3 --port COM5 -b 460800 `
  --before default_reset --after hard_reset `
  write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m `
  0x0     bootloader.bin `
  0x8000  partition-table.bin `
  0xf000  ota_data_initial.bin `
  0x20000 esp32-csi-node.bin
```

Setelah flash, **harus provision ulang** (lihat bagian Ganti WiFi di atas).

> ⚠️ Jika error `Failed to connect`: tahan tombol **BOOT** di ESP32, tekan **RESET**, lepas RESET dulu baru lepas BOOT.

---

## 🔥 Firewall (Sekali Saja per PC)

```powershell
New-NetFirewallRule -DisplayName "WeView ESP32 UDP" `
  -Direction Inbound -Protocol UDP -LocalPort 5005 `
  -Action Allow -Profile Any
```

---

## ❗ Troubleshooting

| Masalah | Solusi |
|---|---|
| ESP32 tidak terdeteksi | Install driver CH343: [wch.cn](http://www.wch.cn/downloads/CH343SER_ZIP.html) |
| `Failed to connect` saat flash | Tahan BOOT + tekan RESET |
| Node tidak kirim data | Cek IP (`ipconfig`), cek firewall, cek channel WiFi |
| Server tidak terima frame | Pastikan server jalan dulu sebelum power on ESP32 |
| `npm not found` di WSL | `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh \| bash && source ~/.bashrc && nvm install 20` |
| Cargo error di Windows | Jalankan `cargo` dari WSL Ubuntu, bukan Git Bash |
| Deprecated warning esptool | Aman, abaikan saja |

---

## 📋 Production Day Checklist

```
□ Laptop charged + charger dibawa
□ 3 power bank charged (min 5000mAh)
□ 3 kabel USB-C
□ Hubungkan laptop ke WiFi venue
□ Catat: SSID=______ Password=______ IP=______ Channel=______
□ Sambungkan 3 ESP32 ke laptop
□ Edit & jalankan reprovision-all.ps1
□ Cabut USB, colok power bank
□ Tempatkan node segitiga di ruangan
□ Jalankan server
□ Buka dashboard, verifikasi 3/3 online
□ Demo: 0→1→2→3 orang, through-wall, camera fusion
```
