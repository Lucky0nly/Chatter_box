"""
Chatterbox TTS - IPC Trigger Client
Sends a READ command to the running reader_bot.py via local socket.

Usage:
    python read_trigger.py

This script is designed to be called from Windows Explorer right-click menu.
"""

import socket
import json
import sys

HOST = "127.0.0.1"
PORT = 5678

try:
    with socket.create_connection((HOST, PORT), timeout=2) as sock:
        message = {"action": "read"}
        sock.sendall(json.dumps(message).encode("utf-8"))
        print("[Trigger] READ command sent.")
except ConnectionRefusedError:
    print("[Error] Chatterbox Reader Bot is not running.")
    print("Start it first with: run_bot.bat")
    sys.exit(1)
except Exception as e:
    print(f"[Error] {e}")
    sys.exit(1)
