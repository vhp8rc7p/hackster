import time

from pymycobot.mypalletizer260 import MyPalletizer260
import time

import RPi.GPIO as GPIO
# PI版本
arm = MyPalletizer260("/dev/ttyAMA0", 1000000)
GPIO.setmode(GPIO.BCM)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(21, GPIO.OUT)


def pump_on():
    GPIO.output(20, 0)
    GPIO.output(21, 0)
# 停止吸泵 m5
def pump_off():
    GPIO.output(20, 1)
    GPIO.output(21, 1)


# 1. Your specific 9 coordinates [x, y, z, yaw]
# Organized in a 0-8 index (Top-Left to Bottom-Right)
TIC_TAC_TOE_COORDS = [
    [180,  15, 100,  10],   # Cell 0 (Top Left)
    [180,   0, 100, 10],  # Cell 1 (Top Center)
    [180, -55, 100, 10],  # Cell 2 (Top Right)
    [130,  25, 100,  10],  # Cell 3 (Middle Left)
    [130,   0, 100, 10],  # Cell 4 (Middle Center)
    [130, -55, 102.3,  10],  # Cell 5 (Middle Right)
    [80,   25,  100, 10],  # Cell 6 (Bottom Left)
    [80,   0,  100, 10],  # Cell 7 (Bottom Center)
    [80,  -55,  100, 10]   # Cell 8 (Bottom Right)
]
PICK_COORD=[-10, -130, 65, 10]
PICK_COORD_ABOVE=[-10, -130, 100, 10]

def reliable_send_coords(coords, speed, timeout=10):
    """
    Sends coordinates and waits until the arm arrives or times out.
    """
    # 2. Send the command
    arm.send_coords(coords, speed)
    
    # 3. Wait for arrival
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = arm.is_in_position(coords, 1) # 1 = check coordinates
        
        if status == 1:
            print(f"Reached position: {coords}")
            return True
        elif status == -1:
            print("Error: Robot reported an invalid state.")
            return False
        
        time.sleep(0.2) # Small sleep to not overwhelm the CPU/Serial
        
    print(f"Timeout: Arm did not reach {coords} within {timeout}s")
    return False
def move_and_wait(coords, speed, timeout=12):
    """
    Sends coords and waits for the robot to stop moving.
    """
    # 1. Send the movement command
    arm.send_coords(coords, speed)
    
    # 2. CRITICAL: Wait a moment for the robot to actually start moving
    # If we check too fast, is_moving() will be 0 because it hasn't 'woken up' yet
    time.sleep(0.5) 
    
    start_time = time.time()
    
    # 3. Poll is_moving() until it returns 0 (stopped)
    while True:
        try:
            moving_status = arm.is_moving()
            
            if moving_status == 0: # Robot has arrived and stopped
                print(f"Movement finished: {coords}")
                break
            elif moving_status == -1:
                print("Robot reported an error state.")
                break
                
        except Exception as e:
            print(f"Serial communication hiccup: {e}")
            
        # 4. Safety Timeout
        if time.time() - start_time > timeout:
            print("Move timed out!")
            break
            
        # 5. DO NOT REMOVE: This prevents the serial port from crashing
        time.sleep(0.2) 

    # 6. Small final settle time
    time.sleep(0.2)
def pick_and_place(cell_index):
    """
    1. Go to Pickup Above -> Down -> SUCK
    2. Go to Pickup Above -> Travel to Cell Above
    3. Go to Cell Down -> RELEASE
    4. Go to Cell Above -> Return to Pickup Above
    """
    target = TIC_TAC_TOE_COORDS[cell_index]
    tx, ty, tz, tyaw = target

    # --- PHASE 1: THE PICK ---
    print("Moving to Pickup Position...")
    arm.send_coords(PICK_COORD_ABOVE, 40) # Hover over supply
    time.sleep(3)
    arm.send_coords(PICK_COORD, 20)       # Move down to touch piece
    time.sleep(3)
    
    pump_on()                             # <--- PUMP ACTIVATED
    time.sleep(1.5)                       # Wait for vacuum to seal
    
    arm.send_coords(PICK_COORD_ABOVE, 40) # Lift the piece up
    time.sleep(5)

    # --- PHASE 2: THE TRAVEL ---
    print(f"Traveling to Cell {cell_index}...")
    # # Hover 40mm above the target cell to avoid hitting other pieces
    # arm.send_coords([tx, ty, tz + 40, tyaw], 40) 
    # time.sleep(1.5)

    # --- PHASE 3: THE PLACE ---
    arm.send_coords([tx, ty, tz, tyaw], 20) # Lower to the board
    time.sleep(5)
    
    pump_off()                            # <--- PUMP DEACTIVATED
    time.sleep(1)                         # Wait for piece to settle
    
    # --- PHASE 4: THE RESET ---
    # arm.send_coords([tx, ty, tz + 40, tyaw], 40) # Lift arm away from piece
    # time.sleep(1)
    
    arm.send_coords(PICK_COORD_ABOVE, 40) # Go back to "Home" pick position
    time.sleep(5)
    print("Ready for next move.")

# --- Test Loop ---
if __name__ == "__main__":
    print("Starting Tic-Tac-Toe Grid Test...")
    for i in range(9):
        user_input = input(f"Press Enter to test Cell {i} (Coord: {TIC_TAC_TOE_COORDS[i]}) or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        pick_and_place(i)

    print("Test complete.")
