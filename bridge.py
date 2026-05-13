"""
WeView ESP32 UDP Bridge
Forwards CSI packets from Windows Hotspot (192.168.137.1:5005)
to WSL Rust server (WSL_IP:5006)

Uses TWO sockets:
  recv_sock → bound to 192.168.137.1:5005 (receives from ESP32)
  send_sock → unbound, sends to WSL_IP:5006 (forwards to WSL)
"""
import socket, sys, subprocess

LISTEN_IP   = "192.168.137.1"
LISTEN_PORT = 5005
TARGET_PORT = 5006

# Auto-detect WSL IP
try:
    result = subprocess.run(["wsl", "-e", "hostname", "-I"], capture_output=True, text=True)
    TARGET_IP = result.stdout.strip().split()[0]
except Exception:
    TARGET_IP = "127.0.0.1"

try:
    # Socket 1: receive from ESP32 via Hotspot interface
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((LISTEN_IP, LISTEN_PORT))

    # Socket 2: send to WSL (unbound, uses default interface)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"✅ Python Bridge Ready! {LISTEN_IP}:{LISTEN_PORT} → {TARGET_IP}:{TARGET_PORT}")
    c = 0
    while True:
        data, addr = recv_sock.recvfrom(4096)
        send_sock.sendto(data, (TARGET_IP, TARGET_PORT))
        c += 1
        if c % 50 == 0:
            print(f"🔥 {c} packets forwarded from {addr[0]}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)
