#!/bin/bash
# rebrand.sh - Script untuk me-rebrand dari WeView ke WeView

echo "Memulai proses rebranding dari WeView ke WeView..."

# Mengganti teks WeView -> WeView (Case Sensitive) di semua file (kecuali folder sistem/build)
find . -type f \
  -not -path "*/\.git/*" \
  -not -path "*/node_modules/*" \
  -not -path "*/target/*" \
  -not -path "*/build/*" \
  -not -path "*/release_bins/*" \
  -exec sed -i 's/WeView/WeView/g' {} +

# Mengganti teks weview -> weview (Case Sensitive)
find . -type f \
  -not -path "*/\.git/*" \
  -not -path "*/node_modules/*" \
  -not -path "*/target/*" \
  -not -path "*/build/*" \
  -not -path "*/release_bins/*" \
  -exec sed -i 's/weview/weview/g' {} +

echo "✅ Rebranding teks selesai!"
echo "Catatan: Jika Anda ingin mengubah nama folder utama dari ~/WeView menjadi ~/WeView,"
echo "silakan jalankan perintah berikut di terminal WSL:"
echo "cd ~ && mv WeView WeView"
