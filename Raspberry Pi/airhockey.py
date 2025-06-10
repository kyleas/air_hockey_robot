#!/usr/bin/env python3
## @file airhockey.py
## @brief Complete Python script for air hockey table detection and control
## @details This script provides:
##  - Frame calibration
##  - HSV calibration 
##  - Main detection loop 
##  - Exponential smoothing on puck to reduce jitter
##  - Serial communication with air hockey table controller
##  - Multiple prediction modes including aggressive behavior
## @author Kyle Schumacher
## @date 2025
## @version 1.0
##
## Usage:
##   python3 airhockey.py --mode calibrate_frame
##   python3 airhockey.py --mode calibrate_hsv
##   python3 airhockey.py --mode run
##
## Dependencies:
##   sudo apt update
##   sudo apt install python3-pip
##   pip3 install opencv-python numpy pyserial
##
## To autostart on boot, create a systemd service pointing to:
##   ExecStart=/usr/bin/python3 /home/pi/airhockey.py --mode run

import cv2
import numpy as np
import json
import argparse
import os
import math
import serial
import time

# ------------------------------------------------------------------------------
## @name Configuration Constants
## @{
# ------------------------------------------------------------------------------

## @brief File to store frame calibration warp matrix
FRAME_CALIB_FILE = "warp_matrix.json"
## @brief File to store HSV color calibration ranges
HSV_CALIB_FILE   = "hsv_ranges.json"

## @brief Serial port device path for communication with table controller
SERIAL_PORT = "/dev/serial0"
## @brief Baud rate for serial communication
BAUD_RATE   = 115200

## @brief Minimum radius for valid object detection (pixels)
MIN_RADIUS = 15
## @brief Minimum contour area threshold for object detection
## Only consider contours with area at least ~half that of a circle radius MIN_RADIUS
AREA_THRESH = math.pi * (MIN_RADIUS ** 2) * 0.5

## @brief Exponential smoothing factor for puck position filtering (0-1)
SMOOTHING_ALPHA = 0.3
## @brief Velocity threshold for determining if puck is moving significantly
VEL_THRESHOLD   = 2.0

## @brief Number of clicks required per side during frame calibration
CLICKS_PER_SIDE = 2

## @brief HSV hue margin for color calibration
H_MARGIN = 10
## @brief HSV saturation margin for color calibration
S_MARGIN = 10
## @brief HSV value margin for color calibration
V_MARGIN = 10

## @brief Target frame rate for detection loop
FRAME_RATE = 30.0

## @}

# ------------------------------------------------------------------------------
## @name Global Variables
## @{
# ------------------------------------------------------------------------------

## @brief List of mouse click coordinates for frame calibration
clicks        = []
## @brief List of HSV color samples for color calibration
hsv_samples   = []

## @brief Current smoothed puck position (x, y)
smoothed_puck       = None
## @brief Previous smoothed puck position for velocity calculation
prev_smoothed_puck  = None

## @}

# ------------------------------------------------------------------------------
## @name Utility Functions
## @{
# ------------------------------------------------------------------------------

## @brief Calculate line equation from two points
## @param pt1 First point (x, y)
## @param pt2 Second point (x, y)
## @return Tuple (a, b, c) representing line equation ax + by + c = 0
def line_from_two_points(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    a = float(y1 - y2)
    b = float(x2 - x1)
    c = float(x1 * y2 - x2 * y1)
    return (a, b, c)

## @brief Find intersection point of two lines
## @param l1 First line (a, b, c)
## @param l2 Second line (a, b, c)
## @return Intersection point (x, y)
def intersect_lines(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    denom = (a1 * b2 - a2 * b1)
    if abs(denom) < 1e-8:
        return (0.0, 0.0)
    x = (b1 * c2 - b2 * c1) / denom
    y = (c1 * a2 - c2 * a1) / denom
    return (x, y)

## @brief Save perspective transformation matrix to file
## @param filename Output filename
## @param matrix 3x3 transformation matrix
## @param width Transformed image width
## @param height Transformed image height
def save_warp_matrix(filename, matrix, width, height):
    data = {
        "matrix": matrix.tolist(),
        "width": float(width),
        "height": float(height)
    }
    with open(filename, "w") as f:
        json.dump(data, f)

## @brief Load perspective transformation matrix from file
## @param filename Input filename
## @return Tuple (matrix, width, height)
def load_warp_matrix(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    mat = np.array(data["matrix"], dtype=np.float32)
    w   = int(data["width"])
    h   = int(data["height"])
    return mat, w, h

## @brief Save HSV color ranges to file
## @param filename Output filename
## @param hsv_dict Dictionary containing HSV min/max values
def save_hsv_ranges(filename, hsv_dict):
    with open(filename, "w") as f:
        json.dump(hsv_dict, f)

## @brief Load HSV color ranges from file
## @param filename Input filename
## @return Tuple (lower_bound, upper_bound) as numpy arrays
def load_hsv_ranges(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    lower = np.array([data["h_min"], data["s_min"], data["v_min"]], dtype=np.uint8)
    upper = np.array([data["h_max"], data["s_max"], data["v_max"]], dtype=np.uint8)
    return lower, upper

## @brief Mouse callback function for frame calibration
## @param event OpenCV mouse event type
## @param x Mouse x coordinate
## @param y Mouse y coordinate
## @param flags Mouse event flags
## @param param User data parameter
def mouse_callback_frame(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))

## @}

# ------------------------------------------------------------------------------
## @name Calibration Functions
## @{
# ------------------------------------------------------------------------------

## @brief Perform frame calibration to determine perspective transformation
## @details User clicks points on each edge of the air hockey table to define
##          the region of interest and calculate perspective transformation matrix
def calibrate_frame():
    global clicks
    clicks = []
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Ensure Pi camera is enabled.")
        return

    window_name = "Frame Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.setMouseCallback(window_name, mouse_callback_frame)

    side_names = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    side_idx = 0

    print("== FRAME CALIBRATION ==")
    print("Click 2 points on each side:")
    print("  1) TOP edge → 'n'")
    print("  2) RIGHT edge → 'n'")
    print("  3) BOTTOM edge → 'n'")
    print("  4) LEFT edge → 'n'")
    print("Press 'q' to abort.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        vis = frame.copy()
        for (x, y) in clicks:
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)

        cv2.putText(vis,
                    f"Click 2 points on the {side_names[side_idx]} edge",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2)
        cv2.imshow(window_name, vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('n'):
            if len(clicks) < (side_idx + 1) * CLICKS_PER_SIDE:
                print(f"  >> Need {CLICKS_PER_SIDE} points on {side_names[side_idx]}.")
                continue
            side_idx += 1
            if side_idx >= 4:
                break
            print(f"Now click 2 points on the {side_names[side_idx]} edge, then 'n'.")
        elif key == ord('q'):
            print("Calibration aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(clicks) != 8:
        print(f"ERROR: Got {len(clicks)} points, expected 8. Aborting.")
        return

    top_pts    = clicks[0:2]
    right_pts  = clicks[2:4]
    bottom_pts = clicks[4:6]
    left_pts   = clicks[6:8]

    l_top    = line_from_two_points(top_pts[0],    top_pts[1])
    l_right  = line_from_two_points(right_pts[0],  right_pts[1])
    l_bottom = line_from_two_points(bottom_pts[0], bottom_pts[1])
    l_left   = line_from_two_points(left_pts[0],   left_pts[1])

    tl = intersect_lines(l_top,    l_left)
    tr = intersect_lines(l_top,    l_right)
    br = intersect_lines(l_bottom, l_right)
    bl = intersect_lines(l_bottom, l_left)

    widthA  = math.hypot(br[0]-bl[0], br[1]-bl[1])
    widthB  = math.hypot(tr[0]-tl[0], tr[1]-tl[1])
    maxWidth  = max(int(widthA), int(widthB))

    heightA = math.hypot(tr[0]-br[0], tr[1]-br[1])
    heightB = math.hypot(tl[0]-bl[0], tl[1]-bl[1])
    maxHeight = max(int(heightA), int(heightB))

    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    save_warp_matrix(FRAME_CALIB_FILE, M, maxWidth, maxHeight)
    print(f"\n[OK] Saved warp matrix to '{FRAME_CALIB_FILE}' ({maxWidth}×{maxHeight}).\n")

## @brief Perform HSV color calibration for object detection
## @details User clicks on objects to sample HSV values and determine
##          appropriate color ranges for thresholding
def calibrate_hsv():
    global hsv_samples
    hsv_samples = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera for HSV calibration.")
        return

    win_raw = "HSV Calibration - Raw"
    win_masked = "HSV Calibration - Masked"
    cv2.namedWindow(win_raw, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_masked, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_raw, 800, 600)
    cv2.resizeWindow(win_masked, 800, 600)

    frame_hsv = None
    ## @brief Mouse callback for HSV calibration
    ## @param event OpenCV mouse event type
    ## @param x Mouse x coordinate
    ## @param y Mouse y coordinate
    ## @param flags Mouse event flags
    ## @param param User data parameter
    def on_mouse(event, x, y, flags, param):
        nonlocal frame_hsv
        if event == cv2.EVENT_LBUTTONDOWN and frame_hsv is not None:
            h, s, v = frame_hsv[y, x]
            hsv_samples.append((int(h), int(s), int(v)))
            print(f"[HSV SAMPLE] ({x},{y}) → H={h}, S={s}, V={v}")

    cv2.setMouseCallback(win_raw, on_mouse)

    print("\n== HSV CALIBRATION ==")
    print("Click each red circle in the RAW window. MASKED shows the mask.")
    print("Press 'q' when done.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if hsv_samples:
            hs = [h for (h,s,v) in hsv_samples]
            ss = [s for (h,s,v) in hsv_samples]
            vs = [v for (h,s,v) in hsv_samples]
            h_min = max(0,   min(hs) - H_MARGIN)
            h_max = min(180, max(hs) + H_MARGIN)
            s_min = max(0,   min(ss) - S_MARGIN)
            s_max = min(255, max(ss) + S_MARGIN)
            v_min = max(0,   min(vs) - V_MARGIN)
            v_max = min(255, max(vs) + V_MARGIN)
        else:
            h_min = h_max = 0
            s_min = s_max = 0
            v_min = v_max = 0

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        vis_raw = frame.copy()
        cv2.putText(vis_raw,
                    f"Samples={len(hsv_samples)}  H=[{h_min}-{h_max}]  S=[{s_min}-{s_max}]  V=[{v_min}-{v_max}]",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2)
        cv2.imshow(win_raw, vis_raw)

        mask = cv2.inRange(frame_hsv, lower, upper)
        masked_vis = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.putText(masked_vis,
                    "Masked Preview",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2)
        cv2.imshow(win_masked, masked_vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not hsv_samples:
        print("No HSV samples; aborting.")
        return

    hs = [h for (h,s,v) in hsv_samples]
    ss = [s for (h,s,v) in hsv_samples]
    vs = [v for (h,s,v) in hsv_samples]
    h_min = max(0,   min(hs) - H_MARGIN)
    h_max = min(180, max(hs) + H_MARGIN)
    s_min = max(0,   min(ss) - S_MARGIN)
    s_max = min(255, max(ss) + S_MARGIN)
    v_min = max(0,   min(vs) - V_MARGIN)
    v_max = min(255, max(vs) + V_MARGIN)

    hsv_dict = {
        "h_min": int(h_min),
        "h_max": int(h_max),
        "s_min": int(s_min),
        "s_max": int(s_max),
        "v_min": int(v_min),
        "v_max": int(v_max)
    }
    save_hsv_ranges(HSV_CALIB_FILE, hsv_dict)

    print(f"\n[OK] Saved HSV to '{HSV_CALIB_FILE}': H[{h_min}-{h_max}] S[{s_min}-{s_max}] V[{v_min}-{v_max}]\n")

## @}

# ------------------------------------------------------------------------------
## @name Physics and Prediction Functions
## @{
# ------------------------------------------------------------------------------

## @brief Reflect a velocity vector off a wall
## @param vx X component of velocity
## @param vy Y component of velocity
## @param hit_wall Wall that was hit ("left", "right", "top", "bottom")
## @return Tuple (new_vx, new_vy) with reflected velocity components
def reflect_vector(vx, vy, hit_wall):
    if hit_wall in ("left", "right"):
        return (-vx, vy)
    elif hit_wall in ("top", "bottom"):
        return (vx, -vy)
    else:
        return (vx, vy)

## @brief Calculate the first wall bounce for a moving object
## @param x0 Initial X position
## @param y0 Initial Y position
## @param vx X velocity component
## @param vy Y velocity component
## @param W Table width
## @param H Table height
## @return Tuple (time, x_hit, y_hit, wall) or None if no collision
def compute_first_bounce(x0, y0, vx, vy, W, H):
    candidates = []
    if vx < 0:
        t = (0 - x0) / vx
        if t > 1e-6:
            y_hit = y0 + t * vy
            if 0 <= y_hit <= H:
                candidates.append((t, 0.0, y_hit, "left"))
    if vx > 0:
        t = (W - x0) / vx
        if t > 1e-6:
            y_hit = y0 + t * vy
            if 0 <= y_hit <= H:
                candidates.append((t, float(W), y_hit, "right"))
    if vy < 0:
        t = (0 - y0) / vy
        if t > 1e-6:
            x_hit = x0 + t * vx
            if 0 <= x_hit <= W:
                candidates.append((t, x_hit, 0.0, "top"))
    if vy > 0:
        t = (H - y0) / vy
        if t > 1e-6:
            x_hit = x0 + t * vx
            if 0 <= x_hit <= W:
                candidates.append((t, x_hit, float(H), "bottom"))

    if not candidates:
        return None
    return min(candidates, key=lambda e: e[0])

## @}

# ------------------------------------------------------------------------------
## @name Main Detection and Control
## @{
# ------------------------------------------------------------------------------

## @brief Main detection loop with serial communication and multiple prediction modes
## @details Runs the complete air hockey detection system including:
##          - Camera capture and image processing
##          - Object detection and tracking
##          - Physics-based trajectory prediction
##          - Aggressive behavior for stuck pucks
##          - Serial communication with table controller
##          - Real-time visualization
def main_loop():
    global smoothed_puck, prev_smoothed_puck

    if not os.path.exists(FRAME_CALIB_FILE):
        print(f"ERROR: '{FRAME_CALIB_FILE}' missing. Run --mode calibrate_frame.")
        return
    if not os.path.exists(HSV_CALIB_FILE):
        print(f"ERROR: '{HSV_CALIB_FILE}' missing. Run --mode calibrate_hsv.")
        return

    warp_matrix, TABLE_W, TABLE_H = load_warp_matrix(FRAME_CALIB_FILE)
    hsv_lower, hsv_upper         = load_hsv_ranges(HSV_CALIB_FILE)

    # 20% from top
    ## @brief Normal Y target position for robot (20% from top of table)
    y_target_normal = 0.2 * TABLE_H  # Store the normal Y target position
    ## @brief Current Y target position (can be overridden by aggressive mode)
    y_target = y_target_normal       # Current Y target (can be overridden by aggressive mode)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2.0)
        print(f"[OK] Serial opened on {SERIAL_PORT} @ {BAUD_RATE}")
    except Exception as e:
        print(f"[WARN] Cannot open serial '{SERIAL_PORT}': {e}")
        ser = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera for main loop.")
        return

    win = "AirHockey Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 600)

    print("\n== RUNNING DETECTION: press 'q' to quit ==\n")

    smoothed_puck = None
    prev_smoothed_puck = None

    # FPS counters
    ## @brief Frame counter for FPS calculation
    fps_count = 0
    ## @brief Start time for FPS calculation
    fps_start = time.time()
    ## @brief Current FPS for display
    fps_display = 0.0

    # Hit mode tracking
    ## @brief Flag indicating if hit mode is currently active
    hit_mode_active = False
    ## @brief Timestamp when hit mode was last activated
    hit_mode_start_time = 0
    ## @brief Duration to maintain hit mode after activation (seconds)
    HIT_MODE_DURATION = 1.0  # seconds

    # Aggressive mode tracking
    ## @brief Flag indicating if puck is in robot's half of table
    puck_in_robot_half = False
    ## @brief Timestamp when puck entered robot's half
    puck_in_robot_half_start_time = 0
    ## @brief Time threshold before activating aggressive mode (seconds)
    PUCK_IN_ROBOT_HALF_THRESHOLD = 1.0  # seconds
    ## @brief Flag indicating if aggressive mode is currently active
    aggressive_mode_active = False
    ## @brief Timestamp when aggressive mode was activated
    aggressive_mode_start_time = 0
    ## @brief Maximum duration for aggressive mode (seconds)
    AGGRESSIVE_MODE_DURATION = 15.0  # seconds - increased for longer possible follow-through
    ## @brief Current phase of aggressive mode (0=inactive, 1=positioning, 2=striking, 3=follow-through)
    aggressive_phase = 0  # 0=inactive, 1=positioning, 2=striking, 3=follow-through
    ## @brief Timestamp when current aggressive phase started
    aggressive_phase_start_time = 0
    ## @brief Stored X position from strike phase for follow-through
    last_strike_x_target = None  # Store last strike position for follow-through
    ## @brief Stored Y position from strike phase for follow-through
    last_strike_y_target = None  # Store last strike Y position for follow-through
    
    # Follow-through settings
    ## @brief Maximum duration for follow-through phase (seconds)
    FOLLOW_THROUGH_TIMEOUT = 10.0  # Maximum follow-through seconds
    ## @brief Flag indicating if puck has crossed midline during follow-through
    puck_crossed_midline = False   # Flag to track if puck crossed midline

    ## @brief Main detection and control loop
    while True:
        ## @brief Record start time for performance monitoring
        loop_start = time.time()

        ## @brief Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            continue

        ## @brief Apply perspective transformation to get bird's-eye view of table
        # This corrects for camera angle and gives us a top-down view
        warped = cv2.warpPerspective(frame, warp_matrix, (TABLE_W, TABLE_H))
        
        ## @brief Convert to HSV color space for better color detection
        # HSV is more robust to lighting changes than RGB
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        
        ## @brief Create binary mask using calibrated HSV ranges
        # White pixels indicate detected objects (pucks/paddles)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        ## @brief Apply Gaussian blur to reduce noise in the mask
        # This helps eliminate small false detections
        mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
        
        ## @brief Find contours of detected objects
        # Contours represent the boundaries of detected objects
        contours_data = cv2.findContours(mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[-2]  # works for both OpenCV 3.x and 4.x

        ## @brief Filter contours by minimum area threshold
        # Only keep contours large enough to be real objects (not noise)
        valid = [c for c in contours if cv2.contourArea(c) >= AREA_THRESH]
        
        ## @brief Sort contours by area (largest first)
        # Largest objects are most likely to be the puck and paddle
        valid.sort(key=lambda c: cv2.contourArea(c), reverse=True)

        ## @brief Create visualization image for debugging and display
        vis = warped.copy()
        
        ## @brief Initialize detection flags and prediction variables
        handle_present = False      # True if paddle/handle detected
        puck_present = False       # True if puck detected
        x_target = None            # Predicted X position for robot to move to
        time_until_impact = None   # Predicted time until puck reaches target line

        ## @brief Object detection and classification logic
        if len(valid) >= 1:
            ## @brief Handle case with two or more objects detected
            if len(valid) >= 2:
                ## @brief Calculate centroids of the two largest objects
                c0 = valid[0]  # Largest contour
                c1 = valid[1]  # Second largest contour
                M0 = cv2.moments(c0)
                M1 = cv2.moments(c1)
                
                ## @brief Verify both objects have valid centroids
                if M0["m00"] > 0 and M1["m00"] > 0:
                    ## @brief Calculate centroid coordinates
                    cx0 = M0["m10"] / M0["m00"]
                    cy0 = M0["m01"] / M0["m00"]
                    cx1 = M1["m10"] / M1["m00"]
                    cy1 = M1["m01"] / M1["m00"]
                    
                    ## @brief Classify objects based on Y position
                    # Object closer to robot (smaller Y) is likely the puck
                    # Object farther from robot (larger Y) is likely the handle/paddle
                    if cy0 < cy1:
                        puck_raw = (cx0, cy0)      # Object 0 is puck
                        handle_raw = (cx1, cy1)    # Object 1 is handle
                    else:
                        puck_raw = (cx1, cy1)      # Object 1 is puck
                        handle_raw = (cx0, cy0)    # Object 0 is handle
                    handle_present = True
                else:
                    ## @brief Handle case where one centroid calculation failed
                    handle_raw = None
                    if M0["m00"] > 0:
                        puck_raw = (M0["m10"] / M0["m00"], M0["m01"] / M0["m00"])
                    elif M1["m00"] > 0:
                        puck_raw = (M1["m10"] / M1["m00"], M1["m01"] / M1["m00"])
                    else:
                        puck_raw = (0.0, 0.0)
                    puck_present = True
            else:
                ## @brief Handle case with single object detected
                c = valid[0]
                M = cv2.moments(c)
                if M["m00"] > 0:
                    puck_raw = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                else:
                    puck_raw = (0.0, 0.0)
                handle_raw = None
                puck_present = True

            ## @brief Apply exponential smoothing to puck position
            # This reduces jitter and noise in position measurements
            if smoothed_puck is None:
                # First detection - no smoothing needed
                smoothed_puck = puck_raw
            else:
                # Apply exponential smoothing filter
                # New position = α * raw_position + (1-α) * previous_smooth_position
                smoothed_puck = (
                    SMOOTHING_ALPHA * puck_raw[0] + (1 - SMOOTHING_ALPHA) * smoothed_puck[0],
                    SMOOTHING_ALPHA * puck_raw[1] + (1 - SMOOTHING_ALPHA) * smoothed_puck[1]
                )

            ## @brief Calculate puck velocity from position history
            if prev_smoothed_puck is not None:
                # Velocity = change in position per frame
                vx = smoothed_puck[0] - prev_smoothed_puck[0]
                vy = smoothed_puck[1] - prev_smoothed_puck[1]
            else:
                # No previous position available
                vx, vy = 0.0, 0.0
            
            ## @brief Store current position for next frame's velocity calculation
            prev_smoothed_puck = smoothed_puck

            ## @brief Draw puck position on visualization
            xp, yp = int(round(smoothed_puck[0])), int(round(smoothed_puck[1]))
            cv2.circle(vis, (xp, yp), 6, (255, 255, 0), -1)  # Cyan dot for puck

            ## @brief Two-object prediction mode (puck + handle detected)
            if handle_present:
                ## @brief Draw handle position on visualization
                xh, yh = int(round(handle_raw[0])), int(round(handle_raw[1]))
                cv2.circle(vis, (xh, yh), 6, (0, 255, 0), -1)  # Green dot for handle

                ## @brief Draw vector from handle to puck
                cv2.line(vis, (xh, yh), (xp, yp), (0, 255, 255), 2)  # Yellow line

                ## @brief Check if puck has sufficient velocity for physics prediction
                puck_vel_mag = math.hypot(vx, vy)
                use_puck_velocity = puck_vel_mag > VEL_THRESHOLD

                if use_puck_velocity:
                    ## @brief Use physics-based prediction with puck velocity
                    x0, y0 = smoothed_puck
                    
                    ## @brief Calculate potential wall bounce
                    fb = compute_first_bounce(x0, y0, vx, vy, TABLE_W, TABLE_H)
                    
                    ## @brief Calculate direct path to target line
                    if abs(vy) > 1e-3:
                        t_direct = (y_target - y0) / vy
                    else:
                        t_direct = None

                    ## @brief Determine if bounce occurs before reaching target
                    need_bounce = False
                    if fb is not None and t_direct is not None:
                        t1, xh1, yh1, w1 = fb
                        if t1 > 0 and t1 < t_direct:
                            need_bounce = True

                    if need_bounce:
                        ## @brief Handle trajectory with wall bounce
                        t1, xh1, yh1, w1 = fb
                        
                        ## @brief Draw path to bounce point
                        cv2.line(vis, (xp, yp), 
                               (int(round(xh1)), int(round(yh1))), 
                               (0, 255, 255), 2)  # Yellow line to bounce
                        cv2.circle(vis, (int(round(xh1)), int(round(yh1))), 
                                 6, (255, 0, 0), -1)  # Blue dot at bounce

                        ## @brief Calculate reflected velocity after bounce
                        vx2, vy2 = reflect_vector(vx, vy, w1)
                        
                        ## @brief Small offset to avoid numerical issues at wall
                        eps = 1e-3
                        x1 = xh1 + vx2 * eps
                        y1 = yh1 + vy2 * eps
                        
                        ## @brief Calculate time from bounce to target line
                        if abs(vy2) > 1e-3:
                            t2 = (y_target - y1) / vy2
                        else:
                            t2 = None

                        if t2 is not None and t2 > 0:
                            ## @brief Calculate final target position after bounce
                            x_target = x1 + vx2 * t2
                            
                            ## @brief Draw path from bounce to target
                            cv2.line(vis, (int(round(xh1)), int(round(yh1))),
                                   (int(round(x_target)), int(round(y_target))),
                                   (255, 0, 255), 2)  # Magenta line after bounce
                            cv2.circle(vis, (int(round(x_target)), int(round(y_target))),
                                     6, (0, 0, 255), -1)  # Red dot at target
                            
                            ## @brief Calculate total time until impact
                            time_until_impact = (t1 + t2) / FRAME_RATE
                        else:
                            ## @brief Handle direct trajectory (no bounce)
                            if t_direct is not None and t_direct > 0:
                                ## @brief Calculate direct target position
                                x_target = x0 + vx * t_direct
                                
                                ## @brief Draw direct path to target
                                cv2.line(vis, (xp, yp),
                                       (int(round(x_target)), int(round(y_target))),
                                       (0, 255, 255), 2)  # Yellow direct line
                                cv2.circle(vis, (int(round(x_target)), int(round(y_target))),
                                         6, (0, 0, 255), -1)  # Red dot at target
                                
                                ## @brief Calculate time until impact
                                time_until_impact = t_direct / FRAME_RATE
                    else:
                        ## @brief Use handle-to-puck vector prediction (low velocity case)
                        # When puck isn't moving much, predict based on handle direction
                        x0, y0 = smoothed_puck
                        
                        ## @brief Calculate vector from handle to puck
                        vx_hp = x0 - handle_raw[0]
                        vy_hp = y0 - handle_raw[1]

                        ## @brief Calculate intersection with target line
                        if abs(vy_hp) > 1e-3:
                            t_direct = (y_target - y0) / vy_hp
                        else:
                            t_direct = None

                        ## @brief Check for wall bounce along handle-puck vector
                        fb = compute_first_bounce(x0, y0, vx_hp, vy_hp, TABLE_W, TABLE_H)
                        need_bounce = False
                        if fb is not None and t_direct is not None:
                            t1, xh1, yh1, w1 = fb
                            if t1 > 0 and t1 < t_direct:
                                need_bounce = True

                        if need_bounce:
                            ## @brief Handle bounce case for handle-puck vector
                            t1, xh1, yh1, w1 = fb
                            
                            ## @brief Draw path to bounce point
                            cv2.line(vis, (xp, yp),
                                   (int(round(xh1)), int(round(yh1))), 
                                   (0, 255, 255), 2)
                            cv2.circle(vis, (int(round(xh1)), int(round(yh1))),
                                     6, (255, 0, 0), -1)  # Blue dot at bounce

                            ## @brief Calculate reflection and final target
                            vx2, vy2 = reflect_vector(vx_hp, vy_hp, w1)
                            eps = 1e-3
                            x1 = xh1 + vx2 * eps
                            y1 = yh1 + vy2 * eps

                            if abs(vy2) > 1e-3:
                                t2 = (y_target - y1) / vy2
                            else:
                                t2 = None

                            if t2 is not None and t2 > 0:
                                x_target = x1 + vx2 * t2
                                cv2.line(vis, (int(round(xh1)), int(round(yh1))),
                                       (int(round(x_target)), int(round(y_target))),
                                       (255, 0, 255), 2)  # Magenta
                                cv2.circle(vis, (int(round(x_target)), int(round(y_target))),
                                         6, (0, 0, 255), -1)  # Red
                        else:
                            ## @brief Direct path case for handle-puck vector
                            if t_direct is not None:
                                x_target = x0 + vx_hp * t_direct
                            else:
                                x_target = x0
                            cv2.line(vis, (xp, yp),
                                   (int(round(x_target)), int(round(y_target))),
                                   (0, 255, 255), 2)  # Yellow
                            cv2.circle(vis, (int(round(x_target)), int(round(y_target))),
                                     6, (0, 0, 255), -1)  # Red

            else:
                ## @brief Single-puck prediction mode (only puck detected)
                mag = math.hypot(vx, vy)
                if (mag > VEL_THRESHOLD):
                    ## @brief Use physics prediction only if puck has significant velocity
                    x0, y0 = smoothed_puck
                    
                    ## @brief Calculate potential wall bounce
                    fb = compute_first_bounce(x0, y0, vx, vy, TABLE_W, TABLE_H)
                    
                    ## @brief Calculate direct path time
                    if abs(vy) > 1e-3:
                        t_direct = (y_target - y0) / vy
                    else:
                        t_direct = None

                    ## @brief Check if bounce occurs before target
                    need_bounce = False
                    if fb is not None and t_direct is not None:
                        t1, xh1, yh1, w1 = fb
                        if t1 > 0 and t1 < t_direct:
                            need_bounce = True

                    if need_bounce:
                        ## @brief Handle bounce trajectory for single puck
                        t1, xh1, yh1, w1 = fb
                        cv2.line(vis, (xp, yp),
                               (int(round(xh1)), int(round(yh1))),
                               (0, 255, 255), 2)  # Yellow to bounce
                        cv2.circle(vis, (int(round(xh1)), int(round(yh1))),
                                 6, (255, 0, 0), -1)  # Blue at bounce

                        vx2, vy2 = reflect_vector(vx, vy, w1)
                        eps = 1e-3
                        x1 = xh1 + vx2 * eps
                        y1 = yh1 + vy2 * eps
                        if abs(vy2) > 1e-3:
                            t2 = (y_target - y1) / vy2
                        else:
                            t2 = None

                        if t2 is not None and t2 > 0:
                            x_target = x1 + vx2 * t2
                            cv2.line(vis, (int(round(xh1)), int(round(yh1))),
                                   (int(round(x_target)), int(round(y_target))),
                                   (255, 0, 255), 2)  # Magenta after bounce
                            cv2.circle(vis, (int(round(x_target)), int(round(y_target))),
                                     6, (0, 0, 255), -1)  # Red at target
                            time_until_impact = (t1 + t2) / FRAME_RATE
                        else:
                            ## @brief Handle direct trajectory for single puck
                            if t_direct is not None and t_direct > 0:
                                x_target = x0 + vx * t_direct
                                cv2.line(vis, (xp, yp),
                                       (int(round(x_target)), int(round(y_target))),
                                       (0, 255, 255), 2)  # Yellow direct
                                cv2.circle(vis, (int(round(x_target)), int(round(y_target))),
                                         6, (0, 0, 255), -1)  # Red at target
                                time_until_impact = t_direct / FRAME_RATE
                    else:
                        ## @brief No prediction when puck velocity is too low
                        # Avoid making predictions when puck is stationary or moving very slowly
                        x_target = None
                        time_until_impact = None

        ## @brief Aggressive behavior state machine for stuck pucks
        # Check if puck is in robot's half (top half) and update timer
        if puck_present and smoothed_puck:
            halfway_y = TABLE_H / 2.0
            current_time = time.time()
            
            ## @brief Track if puck crosses midline during follow-through
            if aggressive_mode_active and aggressive_phase == 3:
                if smoothed_puck[1] > halfway_y:
                    puck_crossed_midline = True
            
            ## @brief Monitor puck position relative to table center
            if smoothed_puck[1] < halfway_y:
                ## @brief Puck is in robot's territory (top half)
                if not puck_in_robot_half:
                    ## @brief Puck just entered robot's half - start timer
                    puck_in_robot_half = True
                    puck_in_robot_half_start_time = current_time
                    puck_crossed_midline = False  # Reset crossing flag
                elif not aggressive_mode_active and (current_time - puck_in_robot_half_start_time) > PUCK_IN_ROBOT_HALF_THRESHOLD:
                    ## @brief Puck stuck in robot's half - activate aggressive mode
                    aggressive_mode_active = True
                    aggressive_mode_start_time = current_time
                    aggressive_phase = 1  # Start with positioning phase
                    aggressive_phase_start_time = current_time
                    puck_crossed_midline = False
                    print("AGGRESSIVE MODE ACTIVATED - POSITIONING PHASE")
            else:
                ## @brief Puck is in human's territory (bottom half) - reset timer
                puck_in_robot_half = False
            
            ## @brief Aggressive mode phase transitions
            if aggressive_mode_active:
                ## @brief Phase 1 → Phase 2: Positioning → Striking
                if aggressive_phase == 1 and (current_time - aggressive_phase_start_time) > 1.0:
                    aggressive_phase = 2
                    aggressive_phase_start_time = current_time
                    print("AGGRESSIVE MODE - STRIKING PHASE")
                ## @brief Phase 2 → Phase 3: Striking → Follow-through
                elif aggressive_phase == 2 and (current_time - aggressive_phase_start_time) > 0.2:
                    aggressive_phase = 3
                    aggressive_phase_start_time = current_time
                    puck_crossed_midline = False
                    print("AGGRESSIVE MODE - FOLLOW-THROUGH PHASE (until puck crosses midline)")
                ## @brief Phase 3 → End: Follow-through → Normal operation
                elif aggressive_phase == 3 and (puck_crossed_midline or 
                                             (current_time - aggressive_phase_start_time) > FOLLOW_THROUGH_TIMEOUT):
                    aggressive_mode_active = False
                    aggressive_phase = 0
                    y_target = y_target_normal  # Reset Y position to normal
                    if puck_crossed_midline:
                        print("Aggressive mode complete - puck crossed midline, returning to normal Y position")
                    else:
                        print("Aggressive mode complete - follow-through timeout (10s), returning to normal Y position")
                
                ## @brief Override normal prediction with aggressive behavior
                if aggressive_mode_active and puck_present:
                    ## @brief Define target goal for aggressive shot
                    goal_x = TABLE_W / 2.0  # Center of opponent's goal
                    goal_y = TABLE_H        # Bottom of table (opponent's end)
                    
                    ## @brief Calculate vector from puck to goal
                    puck_x, puck_y = smoothed_puck
                    goal_vector_x = goal_x - puck_x
                    goal_vector_y = goal_y - puck_y
                    
                    ## @brief Normalize the goal vector
                    vector_length = math.hypot(goal_vector_x, goal_vector_y)
                    if vector_length > 1e-3:
                        norm_vector_x = goal_vector_x / vector_length
                        norm_vector_y = goal_vector_y / vector_length
                        
                        ## @brief Phase 1: Position at intercept point
                        if aggressive_phase == 1:
                            ## @brief Find where puck-to-goal vector crosses robot's Y line
                            if abs(norm_vector_y) > 1e-3:  # Avoid division by zero
                                t = (y_target_normal - puck_y) / norm_vector_y
                                x_target = puck_x + norm_vector_x * t
                                x_target = max(0, min(x_target, TABLE_W))  # Keep in bounds
                                time_until_impact = None  # Not striking yet
                            else:
                                ## @brief Handle horizontal vectors
                                x_target = puck_x
                                time_until_impact = None
                        
                        ## @brief Phase 2: Strike toward halfway point
                        elif aggressive_phase == 2:
                            ## @brief Calculate strike point to reach table center
                            halfway_point = TABLE_H / 2.0
                            
                            if abs(norm_vector_y) > 1e-3:
                                ## @brief Find where vector intersects halfway line
                                t_halfway = (halfway_point - puck_y) / norm_vector_y
                                
                                ## @brief Calculate target position at halfway point
                                x_target = puck_x + norm_vector_x * t_halfway
                                strike_y_target = puck_y + norm_vector_y * t_halfway
                                
                                ## @brief Ensure targets stay within table bounds
                                x_target = max(0, min(x_target, TABLE_W))
                                strike_y_target = max(0, min(strike_y_target, TABLE_H))
                                
                                time_until_impact = 0.2  # Quick strike movement
                                
                                ## @brief Store positions for follow-through phase
                                last_strike_x_target = x_target
                                last_strike_y_target = strike_y_target
                            else:
                                ## @brief Handle horizontal vectors - strike toward center
                                x_target = TABLE_W / 2.0
                                strike_y_target = halfway_point
                                time_until_impact = 0.2
                                last_strike_x_target = x_target
                                last_strike_y_target = strike_y_target
                        
                        ## @brief Phase 3: Follow through - maintain strike position
                        elif aggressive_phase == 3 and last_strike_x_target is not None:
                            ## @brief Hold extended position for momentum and power
                            x_target = last_strike_x_target
                            y_target = last_strike_y_target  # Override normal Y position
                            time_until_impact = None  # No timing needed for follow-through
                        
                        ## @brief Visualization for aggressive mode
                        if aggressive_phase == 1:
                            mode_text = "POSITION"
                        elif aggressive_phase == 2:
                            mode_text = "STRIKE"
                        else:
                            mode_text = "FOLLOW"
                        
                        ## @brief Draw puck-to-goal vector
                        cv2.line(vis, (int(round(puck_x)), int(round(puck_y))),
                               (int(round(goal_x)), int(round(goal_y))),
                               (255, 0, 255), 1)  # Thin magenta line to goal
                        
                        ## @brief Draw robot target position
                        cv2.line(vis, (int(round(puck_x)), int(round(puck_y))),
                               (int(round(x_target)), int(round(y_target))),
                               (0, 0, 255), 3)  # Thick red line for aggressive target
                        cv2.circle(vis, (int(round(x_target)), int(round(y_target))),
                                 8, (0, 0, 255), -1)  # Large red dot
                        
                        ## @brief Disable normal hit mode during aggressive behavior
                        hit_mode_active = False

        ## @brief Update FPS counter for performance monitoring
        fps_count += 1
        now = time.time()
        elapsed = now - fps_start
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count = 0
            fps_start = now
            
        ## @brief Hit mode state management
        # Send command over serial on every frame if we have a valid target
        if x_target is not None and ser is not None:
            try:
                ## @brief Determine if hit mode should be activated
                hit_mode_trigger = (time_until_impact is not None and time_until_impact < 0.4) or \
                            (puck_present and smoothed_puck and abs(smoothed_puck[1] - y_target) < TABLE_H * 0.15)
                
                current_time = time.time()
                
                ## @brief Activate hit mode and set timer
                if hit_mode_trigger:
                    hit_mode_active = True
                    hit_mode_start_time = current_time
                
                ## @brief Check if hit mode should expire
                if hit_mode_active and (current_time - hit_mode_start_time) > HIT_MODE_DURATION:
                    hit_mode_active = False
                
                ## @brief Use hit mode state for position adjustment
                in_hit_mode = hit_mode_active
                
                ## @brief Calculate table coordinate percentages
                table_width = float(TABLE_W)
                adjusted_x_target = x_target
                
                ## @brief Y target adjustment for hit mode
                y_target_adder = 0.0
                if in_hit_mode:
                    # Move slightly forward during hit mode for better contact
                    y_target_adder = 0.05 * TABLE_H

                ## @brief Convert to percentage coordinates (0.0 to 1.0)
                percent_x = x_target / TABLE_W
                percent_y = (y_target + y_target_adder) / TABLE_H

                ## @brief Clamp percentages to valid range
                percent_x = max(percent_x, 0.0)
                percent_x = min(percent_x, 1.0)
                percent_y = max(percent_y, 0.0)
                percent_y = min(percent_y, 1.0)

                ## @brief Scale to controller coordinate system
                # Controller expects coordinates scaled to specific ranges
                scaled_x = int((percent_x) * 2857)
                scaled_y = int((percent_y) * 4873)             
                
                ## @brief Format command for serial transmission
                # Command format: MXXXXYYYY where XXXX and YYYY are 4-digit coordinates
                msg = f"M{scaled_x:04d}{scaled_y:04d}\r\n"
                
                ## @brief Send command byte-by-byte with delays
                # Small delays help prevent buffer overruns on the controller
                for byte in msg.encode('ascii'):
                    ser.write(bytes([byte]))  # Send single byte
                    time.sleep(0.001)  # 1ms delay between bytes
            except Exception as e:
                print(f"Error sending command: {e}")
            
        ## @brief Terminal output for monitoring (once per second)
        current_time = time.time()
        if x_target is not None and (not hasattr(main_loop, "last_print_time") or (current_time - main_loop.last_print_time) >= 1.0):
            ## @brief Determine current operational mode
            mode_str = "HIT" if hit_mode_active else "PREDICT"
            status_msg = f"{mode_str}: Target={x_target:.1f},{y_target:.1f}"
            
            ## @brief Add timing information if available
            if time_until_impact is not None:
                status_msg += f" Time={time_until_impact:.2f}s"
            else:
                status_msg += " Time=unknown"
                
            ## @brief Add serial command information
            if ser is not None:
                status_msg += f" Command=M{scaled_x:04d}{scaled_y:04d}"
                
            print(status_msg)
            main_loop.last_print_time = current_time

        ## @brief Display FPS counter on visualization
        cv2.putText(vis, f"FPS: {fps_display:.1f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        ## @brief Display current operational mode
        if aggressive_mode_active:
            if aggressive_phase == 1:
                mode_text = "Aggressive-Position"
                mode_color = (255, 0, 255)  # Magenta for positioning
            elif aggressive_phase == 2:
                mode_text = "Aggressive-Strike"
                mode_color = (0, 0, 255)    # Red for striking
            else:
                mode_text = "Aggressive-Follow"
                mode_color = (255, 165, 0)  # Orange for follow-through
        elif hit_mode_active:
            mode_text = "Hit"
            mode_color = (0, 0, 255)        # Red for hit mode
        else:
            mode_text = "Predict"
            mode_color = (0, 255, 0)        # Green for prediction mode
            
        cv2.putText(vis, mode_text, (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

        ## @brief Display processed image in window
        vis_display = cv2.resize(vis, (800, 600), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win, vis_display)

        ## @brief Check for quit command
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    ## @brief Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()
        print("\nSerial port closed.\n")

## @}

# ------------------------------------------------------------------------------
## @name Program Entry Point
## @{
# ------------------------------------------------------------------------------

## @brief Main program entry point with command line argument parsing
if __name__ == "__main__":
    ## @brief Setup command line argument parser
    parser = argparse.ArgumentParser(description="Air Hockey Table Detection on Raspberry Pi")
    parser.add_argument(
        "--mode",
        choices=["calibrate_frame", "calibrate_hsv", "run"],
        required=True,
        help="Mode = calibrate_frame | calibrate_hsv | run"
    )
    args = parser.parse_args()

    ## @brief Execute requested mode
    if args.mode == "calibrate_frame":
        calibrate_frame()
    elif args.mode == "calibrate_hsv":
        calibrate_hsv()
    elif args.mode == "run":
        main_loop()
    else:
        print("Unknown mode. Use --mode calibrate_frame / calibrate_hsv / run.")

## @}
