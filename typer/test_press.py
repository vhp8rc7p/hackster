import time
from pymycobot import MyPalletizer260

# Initialize Robot
mc = MyPalletizer260("/dev/tty.usbserial-59010016581", 115200)

# --- 1. CALIBRATED CONSTANTS ---
PITCH_Y = 18.0    # Horizontal distance between keys
Z_PRESSED = 17.0  # Your calibrated press height
Z_HOVER = 30.0    # Lifted height
SPEED = 80
def move_wait(coords, speed):
    # Send the movement command
    # mode=0 for linear (straight) movement, mode=1 for joint movement
    mc.send_coords(coords, speed) 
    
    # Wait until the robot reports it is in position
    # We add a tiny sleep inside the loop to avoid flooding the serial port
    while not mc.is_in_position(coords):
        time.sleep(0.1) 
# Key: (Row_Index, Column_Offset_from_G, X_Coordinate, RX_Angle)
# Row Indices: Top = 1, Home = 0, Bottom = -1
key_map = {
    # TOP ROW (X approx 229, RX approx -24.6)
    'q': (1, -4, 230, -24.6), 'w': (1, -3, 230, -24.6), 
    'e': (1, -2, 230, -24.6), 'r': (1, -1, 230, -24.6),
    't': (1, 0, 230, -24.6),  'y': (1, 1, 230, -24.6),
    'u': (1, 2.5, 230, -24.6),  'i': (1, 3.5, 230, -24.6),
    'o': (1, 4.5, 230, -24.6),  'p': (1, 5.5, 230, -24.6),

    # HOME ROW (X approx 215, RX approx -14.94)
    'a': (0, -4.5, 215.0, -14.94), 's': (0, -3.5, 215.0, -14.94),
    'd': (0, -2.5, 215.0, -14.94),   'f': (0, -1, 215.0, -14.94),
    'g': (0, 0, 215.0, -14.94),    'h': (0, 1, 215.0, -14.94),
    'j': (0, 2, 215.0, -14.94),    'k': (0, 3, 215.0, -14.94),
    'l': (0, 4, 215.0, -14.94),

    # BOTTOM ROW (X approx 191, RX approx -24.69)
    'z': (-1, -4, 191.2, -24.69), 'x': (-1, -3, 191.2, -24.69),
    'c': (-1, -2, 191.2, -24.69), 'v': (-1, -1, 191.2, -24.69),
    'b': (-1, 0.2, 191.2, -24.69),    'n': (-1, 1.2, 191.2, -24.69),
    'm': (-1, 2.2, 191.2, -24.69),
    
    # SPECIAL
    ' ': ( -1.5, 0, 175.0, -24.0),
    '\n': (0, 7, 215.0, -14.94),
}

# The "Stagger" offsets (How much Y changes just by switching rows)
# Calculated from your data: T(row1) was at Y=10.2, B(row-1) was at Y=-6.1
ROW_STAGGER = {
    1: 10.2,   # Top row shift
    0: 0.0,    # Home row (our 0 reference)
   -1: -6.1    # Bottom row shift
}

prev_key = None
prev_coords = None

def settle_time(char, target_x, target_y):
    global prev_key, prev_coords
    if prev_key is None:
        return 0.9
    if char == prev_key:
        return 0.1
    if prev_coords is None:
        return 0.9
    dx = target_x - prev_coords[0]
    dy = target_y - prev_coords[1]
    dist = (dx**2 + dy**2) ** 0.5
    # Scale: nearby keys ~0.2s, far keys ~0.9s
    t = 0.1 + (dist / 120.0) * 1.6
    return min(max(t, 0.1), 1.7)

def type_key(char):
    global prev_key, prev_coords
    char = char.lower()
    if char not in key_map:
        print(f"Key '{char}' not mapped.")

    row_idx, col_offset, target_x, target_rx = key_map[char]

    target_y = ROW_STAGGER.get(row_idx, 0) - (col_offset * PITCH_Y)

    wait = settle_time(char, target_x, target_y)
    print(f"Typing '{char}' -> Coords: [{target_x}, {target_y}, {Z_PRESSED}, {target_rx}] (settle: {wait:.2f}s)")

    # 1. Hover above target
    move_wait([target_x, target_y, Z_HOVER, target_rx], 100)
    time.sleep(wait)
    # 2. Press
    move_wait([target_x, target_y, Z_PRESSED, target_rx], 20)

    # 3. Lift
    move_wait([target_x, target_y, Z_HOVER, target_rx], 100)

    prev_key = char
    prev_coords = (target_x, target_y)
# --- EXECUTION ---
# Safe start position


# test_sentence = "asdfghjkl" 
# test_sentence = "qwertyuiop" 
# test_sentence = "zxcvbnm" 
test_sentence = "you gave me the syntax\nbut i provide the pressure\ntogether we bridge the gap\nbetween a thought and mark"
# test_sentence="\n"
import re
test_sentence = re.sub(r'\n+', '\n', test_sentence)
for letter in test_sentence:
    type_key(letter)