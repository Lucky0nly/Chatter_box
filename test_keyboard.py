
import keyboard
import time

print("Testing keyboard...")
try:
    keyboard.add_hotkey('ctrl+alt+t', lambda: print("Hotkey works!"))
    print("Hotkey added. Waiting 5 seconds...")
    time.sleep(5)
    print("Done waiting.")
except Exception as e:
    print(f"Error: {e}")
