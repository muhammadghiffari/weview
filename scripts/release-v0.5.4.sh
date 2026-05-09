#!/bin/bash
# Release script for v0.5.4-esp32
# Run AFTER firmware build completes and all tests pass
#
# Prerequisites:
#   - firmware/esp32-csi-node/build/esp32-csi-node.bin (8MB build)
#   - All Rust tests passing (1,031+)
#   - Python proof VERDICT: PASS
#
# Usage: bash scripts/release-v0.5.4.sh

set -euo pipefail

TAG="v0.5.4-esp32"
BUILD_DIR="firmware/esp32-csi-node/build"
DIST_DIR="dist/${TAG}"

echo "=== Preparing release ${TAG} ==="

# Verify build artifacts exist
for f in \
  "${BUILD_DIR}/esp32-csi-node.bin" \
  "${BUILD_DIR}/bootloader/bootloader.bin" \
  "${BUILD_DIR}/partition_table/partition-table.bin" \
  "${BUILD_DIR}/ota_data_initial.bin"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Missing build artifact: $f"
    echo "Run the firmware build first."
    exit 1
  fi
done

# Create dist directory
mkdir -p "${DIST_DIR}"

# Copy binaries
cp "${BUILD_DIR}/esp32-csi-node.bin" "${DIST_DIR}/"
cp "${BUILD_DIR}/bootloader/bootloader.bin" "${DIST_DIR}/"
cp "${BUILD_DIR}/partition_table/partition-table.bin" "${DIST_DIR}/"
cp "${BUILD_DIR}/ota_data_initial.bin" "${DIST_DIR}/"

# Generate SHA-256 hashes
echo "=== SHA-256 Hashes ==="
cd "${DIST_DIR}"
sha256sum *.bin > SHA256SUMS.txt
cat SHA256SUMS.txt
cd -

# Binary sizes
echo ""
echo "=== Binary Sizes ==="
ls -lh "${DIST_DIR}"/*.bin

echo ""
echo "=== Release artifacts ready in ${DIST_DIR} ==="
echo ""
echo "Next steps:"
echo "  1. Flash to COM9: esptool.py --chip esp32s3 --port COM9 write_flash 0x0 ${DIST_DIR}/bootloader.bin 0x8000 ${DIST_DIR}/partition-table.bin 0xd000 ${DIST_DIR}/ota_data_initial.bin 0x10000 ${DIST_DIR}/esp32-csi-node.bin"
echo "  2. Tag: git tag ${TAG}"
echo "  3. Push: git push origin ${TAG}"
echo "  4. Release: gh release create ${TAG} ${DIST_DIR}/*.bin ${DIST_DIR}/SHA256SUMS.txt --title 'ESP32-S3 CSI Firmware ${TAG} — Cognitum Seed Integration' --notes-file -"
