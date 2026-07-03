import cv2
import cv2.aruco as aruco
import numpy as np


import cv2
import cv2.aruco as aruco
import numpy as np

def get_board_state():
    cap = cv2.VideoCapture(0)
    
    # ArUco Setup (Legacy compatible)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    saved_slots = [] # This will hold our 9 (x, y) coordinates
    calibrated = False

    print("STEP 1: Place all 9 markers on the board.")
    print("Press 's' to SAVE the board layout once ready.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        
        # Display logic
        if not calibrated:
            cv2.putText(frame, "CALIBRATION MODE: Place 9 markers & press 's'", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "GAME MODE: 'q' to quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Logical Board for current frame
        current_board = [[' ' for _ in range(3)] for _ in range(3)]

        if ids is not None:
            # Get centers of all detected markers
            detected_centers = []
            for i in range(len(ids)):
                c = corners[i][0]
                cx, cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                detected_centers.append({'id': ids[i][0], 'pos': (cx, cy)})
                
                # Draw circles on detected markers
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 255), -1)

            # --- CALIBRATION LOGIC ---
            if not calibrated and cv2.waitKey(1) & 0xFF == ord('s'):
                if len(detected_centers) == 9:
                    # Sort by Y (rows)
                    detected_centers.sort(key=lambda x: x['pos'][1])
                    
                    # Sort each row by X (columns)
                    rows = [detected_centers[0:3], detected_centers[3:6], detected_centers[6:9]]
                    for r in rows:
                        r.sort(key=lambda x: x['pos'][0])
                    
                    # Flatten back into a list of 9 coordinates
                    saved_slots = [item['pos'] for r in rows for item in r]
                    calibrated = True
                    print("Board Calibrated! You can now clear the markers and play.")
                else:
                    print(f"Error: Need exactly 9 markers to calibrate. Found {len(detected_centers)}.")

            # --- GAMEPLAY LOGIC ---
            if calibrated:
                for piece in detected_centers:
                    # Find which saved slot is closest to this piece
                    px, py = piece['pos']
                    
                    distances = [np.sqrt((px-sx)**2 + (py-sy)**2) for sx, sy in saved_slots]
                    closest_idx = np.argmin(distances)
                    
                    # Only accept if it's reasonably close (e.g., within 60 pixels)
                    if distances[closest_idx] < 60:
                        row = closest_idx // 3
                        col = closest_idx % 3
                        current_board[row][col] = 'X' if piece['id'] == 10 else 'O'
                        
                        # Visual label
                        cv2.putText(frame, f"Slot {row},{col}", (int(px), int(py)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the remembered "Ghost" slots
        for idx, (sx, sy) in enumerate(saved_slots):
            cv2.circle(frame, (int(sx), int(sy)), 10, (255, 255, 255), 1)

        cv2.imshow('Tic Tac Toe Memory', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return current_board

final = get_board_state()
print("\nFinal Board State:")
for r in final: print(r)
