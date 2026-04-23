import pyautogui as gui
import time

print("Point at the game. Press Ctrl+C to stop.")
try:
    while True:
        x, y = gui.position()
        # Adding 20 spaces at the end to clear old digits
        print(f"X: {x}, Y: {y}                    ", end="\r")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDone.")