import pyautogui
import time

text = """you gave me the syntax
but i provide the pressure
together we bridge the gap
between a thought and mark"""

print("Switch to the browser window. Starting in 3 seconds...")
time.sleep(3)

for ch in text:
    if ch == "\n":
        pyautogui.press("enter")
        time.sleep(1.5)
    else:
        pyautogui.press(ch) if len(ch) == 1 else None
        time.sleep(0.9)

print("Done.")
