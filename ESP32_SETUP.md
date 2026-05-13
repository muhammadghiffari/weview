# WeView — ESP32 3-Node Setup & Operations Guide

> **3× ESP32-S3N16R8** | All nodes flashed & provisioned ✅  
> Last updated: 2026-05-12

---

## 🚀 ALUR PRODUKSI (Production Workflow)

Alur ini dirancang khusus untuk mengatasi isolasi jaringan (AP isolation) yang sering terjadi di WiFi kampus/publik, serta mengatasi bug jaringan WSL2 Mirrored Mode pada Windows.

### STEP 1: Aktifkan Mobile Hotspot Windows
**Jangan gunakan WiFi venue secara langsung untuk ESP32!**
1. Hubungkan laptop ke WiFi venue (untuk akses internet).
2. Nyalakan **Mobile Hotspot** di Windows.
3. Atur nama hotspot: `WeView-Net` (atau sesuai keinginan).
4. Catat IP Hotspot laptop Anda. Biasanya `192.168.137.1` (Cek dengan `ipconfig` di PowerShell pada adapter "Local Area Connection*").

### STEP 2: Nyalakan Server Rust di WSL (Port 5006)
Buka terminal **WSL Ubuntu**, lalu jalankan server:
```bash
cd ~/WeView/v2
cargo run -p wifi-densepose-sensing-server -- --source esp32 --udp-port 5006
```
*(Pastikan muncul tulisan: `UDP listening on 0.0.0.0:5006`)*

### STEP 3: Jalankan Python Bridge di PowerShell
Karena WSL tidak bisa menerima paket UDP langsung dari Hotspot IP, kita butuh "Jembatan".
Buka **PowerShell baru** (jangan tutup WSL), lalu jalankan:
```powershell
python \\wsl.localhost\Ubuntu-22.04\home\kezman\WeView\bridge.py
```
*(Jika error `10048 Only one usage...`, restart laptop Anda untuk mematikan zombie process)*

### STEP 4: Sambungkan ESP32 ke Power
Colokkan ke-3 ESP32 ke **Power Bank**. Tunggu 15-20 detik.
Perhatikan PowerShell yang menjalankan bridge. Jika sukses, akan muncul:
`🔥 50 packets forwarded from ...`
*(Ini artinya data ESP32 sudah masuk ke server Rust!)*

### STEP 5: Kalibrasi Ruangan (PENTING!)
Sebelum UI bisa mendeteksi orang, model matematika harus dikalibrasi.
1. Kosongkan ruangan (jangan ada orang berjalan).
2. Buka **PowerShell baru**, jalankan:
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8080/api/v1/calibration/start"
```
3. **TUNGGU 15 MENIT.** (Dibutuhkan 12,000 frame WiFi agar akurat).
4. Setelah 15 menit, hentikan kalibrasi:
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8080/api/v1/calibration/stop"
```
*(Jika muncul `"success": true`, sistem SIAP DIGUNAKAN!)*

### STEP 6: Buka Dashboard
Buka browser di Windows dan akses:
- **Main Dashboard & 3D Pose:** `http://localhost:8080/ui/index.html`
- **Quantum NVSim Dashboard:** `http://localhost:8080/ui/nvsim/index.html`

---

## 🔄 Ganti WiFi Hotspot (Re-Provisioning)

Jika Anda mengganti nama Hotspot, password, atau ingin mengubah konfigurasi node, ikuti langkah ini **satu per satu untuk setiap node**:

1. Buka PowerShell.
2. Pindah ke direktori firmware:
```powershell
cd \\wsl.localhost\Ubuntu-22.04\home\kezman\WeView\firmware\esp32-csi-node
```
3. Colokkan **Node 1** via USB (misal di COM5), lalu jalankan:
```powershell
python provision.py --port COM5 --ssid WeView-Net --password PASSWORD_HOTSPOT --target-ip 192.168.137.1 --target-port 5005 --node-id 1 --channel 6 --tdm-slot 0 --tdm-total 3 --edge-tier 0
```
4. Cabut Node 1. Colokkan **Node 2** (misal COM3), jalankan dengan `--node-id 2` dan `--tdm-slot 1`.
5. Cabut Node 2. Colokkan **Node 3** (misal COM6), jalankan dengan `--node-id 3` dan `--tdm-slot 2`.

*(Catatan: `--edge-tier 0` Wajib digunakan agar ESP32 mengirim Raw CSI untuk kalibrasi).*

---

## 📐 Penempatan Node
Taruh ESP32 di ketinggian sekitar 1 meter dari lantai (misal di atas kursi, rak, atau tripod kecil), seukuran dengan tinggi pinggul manusia. Jangan ditaruh langsung di lantai.

```
[Meja + Laptop]
        [Node 1]           Bentuk segitiga
         /    \            di dalam ruangan
        /      \           
       /  AREA  \          Jarak ideal: 2-5m
      /  DETEKSI \         Tinggi: ~1m
     /            \        
[Node 2] ——————— [Node 3]
```

---

## Node Registry

| Node | COM Default | MAC Address | Node ID | TDM Slot |
|------|-------------|-------------|---------|----------|
| 🔴 1 | COM5 | `e0:72:a1:d6:db:bc` | 1 | 0/3 |
| 🔵 2 | COM3 | `dc:b4:d9:06:fa:34` | 2 | 1/3 |
| 🟢 3 | COM6 | `14:c1:9f:28:b2:18` | 3 | 2/3 |

All chips: ESP32-S3 (QFN56) rev v0.2, PSRAM 8MB, Flash 16MB
