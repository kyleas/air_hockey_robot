#!/usr/bin/env python3
# airhockey.py
#
# Complete Python script with:
#  1) Frame calibration (windowed, not fullscreen)
#  2) HSV calibration (windowed; live raw + masked preview)
#  3) Main detection loop (fullscreen; centroid‐based detection, optimized)
#  4) Exponential smoothing on puck to reduce jitter
#  5) Drawing:
#       • Two centroids: handle→puck vector, then puck→20% (direct or reflection)
#       • Single puck, low velocity & puck below halfway: vertical line
#       • Single puck, else: puck→20% (direct or reflection)
#       • Small dots at centroids instead of circles around objects
#       • Live FPS counter overlay
#  6) Serial output: "x_target,time_until_impact\n" (only for single‐puck velocity cases)
#
# Usage:
#   python3 airhockey.py --mode calibrate_frame
#   python3 airhockey.py --mode calibrate_hsv
#   python3 airhockey.py --mode run
#
# Dependencies:
#   sudo apt update
#   sudo apt install python3-pip
#   pip3 install opencv-python numpy pyserial
#
# To autostart on boot, create a systemd service pointing to:
#   ExecStart=/usr/bin/python3 /home/pi/airhockey.py --mode run
#

import cv2
import numpy as np
import json
import argparse
import os
import math
import serial
import time

# ------------------------------------------------------------------------------
# CONFIGURABLE CONSTANTS
# ------------------------------------------------------------------------------
FRAME_CALIB_FILE = "warp_matrix.json"
HSV_CALIB_FILE   = "hsv_ranges.json"

SERIAL_PORT = "/dev/serial0"
BAUD_RATE   = 115200

MIN_RADIUS = 15
# Only consider contours with area at least ~half that of a circle radius MIN_RADIUS
AREA_THRESH = math.pi * (MIN_RADIUS ** 2) * 0.5

SMOOTHING_ALPHA = 0.3
VEL_THRESHOLD   = 2.0

CLICKS_PER_SIDE = 2

H_MARGIN = 10
S_MARGIN = 10
V_MARGIN = 10

FRAME_RATE = 30.0

# ------------------------------------------------------------------------------
# GLOBALS FOR MOUSE CALLBACKS & SMOOTHING STATE
# ------------------------------------------------------------------------------
clicks        = []
hsv_samples   = []

smoothed_puck       = None
prev_smoothed_puck  = None

# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------

def line_from_two_points(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    a = float(y1 - y2)
    b = float(x2 - x1)
    c = float(x1 * y2 - x2 * y1)
    return (a, b, c)

def intersect_lines(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    denom = (a1 * b2 - a2 * b1)
    if abs(denom) < 1e-8:
        return (0.0, 0.0)
    x = (b1 * c2 - b2 * c1) / denom
    y = (c1 * a2 - c2 * a1) / denom
    return (x, y)

def save_warp_matrix(filename, matrix, width, height):
    data = {
        "matrix": matrix.tolist(),
        "width": float(width),
        "height": float(height)
    }
    with open(filename, "w") as f:
        json.dump(data, f)

def load_warp_matrix(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    mat = np.array(data["matrix"], dtype=np.float32)
    w   = int(data["width"])
    h   = int(data["height"])
    return mat, w, h

def save_hsv_ranges(filename, hsv_dict):
    with open(filename, "w") as f:
        json.dump(hsv_dict, f)

def load_hsv_ranges(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    lower = np.array([data["h_min"], data["s_min"], data["v_min"]], dtype=np.uint8)
    upper = np.array([data["h_max"], data["s_max"], data["v_max"]], dtype=np.uint8)
    return lower, upper

def mouse_callback_frame(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))

# ------------------------------------------------------------------------------
# CALIBRATION: FRAME WARP
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# CALIBRATION: HSV RANGE (WINDOWED)
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# BOUNCE / REFLECTION LOGIC
# ------------------------------------------------------------------------------

def reflect_vector(vx, vy, hit_wall):
    if hit_wall in ("left", "right"):
        return (-vx, vy)
    elif hit_wall in ("top", "bottom"):
        return (vx, -vy)
    else:
        return (vx, vy)

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

# ------------------------------------------------------------------------------
# MAIN DETECTION + SERIAL SENDING
# ------------------------------------------------------------------------------

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
    y_target = 0.2 * TABLE_H

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
    fps_count = 0
    fps_start = time.time()
    fps_display = 0.0
    
    # Hit mode tracking
    hit_mode_active = False
    hit_mode_start_time = 0
    HIT_MODE_DURATION = 1.0  # seconds
    
    # Aggressive mode tracking
    puck_in_robot_half = False
    puck_in_robot_half_start_time = 0
    PUCK_IN_ROBOT_HALF_THRESHOLD = 1.0  # seconds
    aggressive_mode_active = False
    aggressive_mode_start_time = 0
    AGGRESSIVE_MODE_DURATION = 15.0  # seconds - increased for longer possible follow-through
    aggressive_phase = 0  # 0=inactive, 1=positioning, 2=striking, 3=follow-through
    aggressive_phase_start_time = 0
    last_strike_x_target = None  # Store last strike position for follow-through
    
    # Follow-through settings
    FOLLOW_THROUGH_TIMEOUT = 10.0  # Maximum follow-through seconds
    puck_crossed_midline = False   # Flag to track if puck crossed midline

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        # Warp bird's-eye view
        warped = cv2.warpPerspective(frame, warp_matrix, (TABLE_W, TABLE_H))
        # Convert to HSV and threshold once
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        # Blur the mask to reduce noise then find contours
        mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
        contours_data = cv2.findContours(mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[-2]  # works for both OpenCV 3.x and 4.x

        valid = [c for c in contours if cv2.contourArea(c) >= AREA_THRESH]
        valid.sort(key=lambda c: cv2.contourArea(c), reverse=True)

        vis = warped.copy()
        handle_present = False
        puck_present = False
        x_target = None
        time_until_impact = None

        if len(valid) >= 1:
            if len(valid) >= 2:
                c0 = valid[0]
                c1 = valid[1]
                M0 = cv2.moments(c0)
                M1 = cv2.moments(c1)
                if M0["m00"] > 0 and M1["m00"] > 0:
                    cx0 = M0["m10"] / M0["m00"]
                    cy0 = M0["m01"] / M0["m00"]
                    cx1 = M1["m10"] / M1["m00"]
                    cy1 = M1["m01"] / M1["m00"]
                    # Puck = smaller y (higher on table), handle = larger y
                    if cy0 < cy1:
                        puck_raw = (cx0, cy0)
                        handle_raw = (cx1, cy1)
                    else:
                        puck_raw = (cx1, cy1)
                        handle_raw = (cx0, cy0)
                    handle_present = True
                else:
                    handle_raw = None
                    if M0["m00"] > 0:
                        puck_raw = (M0["m10"] / M0["m00"], M0["m01"] / M0["m00"])
                    elif M1["m00"] > 0:
                        puck_raw = (M1["m10"] / M1["m00"], M1["m01"] / M1["m00"])
                    else:
                        puck_raw = (0.0, 0.0)
                puck_present = True
            else:
                c = valid[0]
                M = cv2.moments(c)
                if M["m00"] > 0:
                    puck_raw = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                else:
                    puck_raw = (0.0, 0.0)
                handle_raw = None
                puck_present = True

            # Exponential smoothing
            if smoothed_puck is None:
                smoothed_puck = puck_raw
            else:
                smoothed_puck = (
                    SMOOTHING_ALPHA * puck_raw[0] + (1 - SMOOTHING_ALPHA) * smoothed_puck[0],
                    SMOOTHING_ALPHA * puck_raw[1] + (1 - SMOOTHING_ALPHA) * smoothed_puck[1]
                )

            # Velocity
            if prev_smoothed_puck is not None:
                vx = smoothed_puck[0] - prev_smoothed_puck[0]
                vy = smoothed_puck[1] - prev_smoothed_puck[1]
            else:
                vx, vy = 0.0, 0.0
            prev_smoothed_puck = smoothed_puck

            xp, yp = int(round(smoothed_puck[0])), int(round(smoothed_puck[1]))
            cv2.circle(vis, (xp, yp), 6, (255, 255, 0), -1)  # puck dot = cyan

            if handle_present:
                xh, yh = int(round(handle_raw[0])), int(round(handle_raw[1]))
                cv2.circle(vis, (xh, yh), 6, (0, 255, 0), -1)  # handle dot = green

                # Draw handle → puck
                cv2.line(vis, (xh, yh), (xp, yp), (0, 255, 255), 2)  # yellow

                # Check puck velocity
                puck_vel_mag = math.hypot(vx, vy)
                use_puck_velocity = puck_vel_mag > VEL_THRESHOLD

                if use_puck_velocity:
                    # Use puck velocity for prediction (like in single-puck mode)
                    x0, y0 = smoothed_puck
                    fb = compute_first_bounce(x0, y0, vx, vy, TABLE_W, TABLE_H)
                    if abs(vy) > 1e-3:
                        t_direct = (y_target - y0) / vy
                    else:
                        t_direct = None

                    need_bounce = False
                    if fb is not None and t_direct is not None:
                        t1, xh1, yh1, w1 = fb
                        if t1 > 0 and t1 < t_direct:
                            need_bounce = True

                    if need_bounce:
                        t1, xh1, yh1, w1 = fb
                        cv2.line(vis,
                                 (xp, yp),
                                 (int(round(xh1)), int(round(yh1))),
                                 (0, 255, 255),
                                 2)  # yellow
                        cv2.circle(vis,
                                   (int(round(xh1)), int(round(yh1))),
                                   6, (255, 0, 0), -1)  # blue

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
                            cv2.line(vis,
                                     (int(round(xh1)), int(round(yh1))),
                                     (int(round(x_target)), int(round(y_target))),
                                     (255, 0, 255),
                                     2)  # magenta
                            cv2.circle(vis,
                                       (int(round(x_target)), int(round(y_target))),
                                       6, (0, 0, 255), -1)  # red
                            time_until_impact = (t1 + t2) / FRAME_RATE
                    else:
                        if t_direct is not None and t_direct > 0:
                            x_target = x0 + vx * t_direct
                            cv2.line(vis,
                                     (xp, yp),
                                     (int(round(x_target)), int(round(y_target))),
                                     (0, 255, 255),
                                     2)  # yellow
                            cv2.circle(vis,
                                       (int(round(x_target)), int(round(y_target))),
                                       6, (0, 0, 255), -1)  # red
                            time_until_impact = t_direct / FRAME_RATE
                else:
                    # Original handle→puck logic
                    # Compute vector from puck along handle→puck
                    x0, y0 = smoothed_puck
                    vx_hp = x0 - handle_raw[0]
                    vy_hp = y0 - handle_raw[1]

                    # Compute time parameter to reach y_target along that vector
                    if abs(vy_hp) > 1e-3:
                        t_direct = (y_target - y0) / vy_hp
                    else:
                        t_direct = None

                    fb = compute_first_bounce(x0, y0, vx_hp, vy_hp, TABLE_W, TABLE_H)
                    need_bounce = False
                    if fb is not None and t_direct is not None:
                        t1, xh1, yh1, w1 = fb
                        if t1 > 0 and t1 < t_direct:
                            need_bounce = True

                    if need_bounce:
                        t1, xh1, yh1, w1 = fb
                        # Draw puck → first bounce
                        cv2.line(vis,
                                 (xp, yp),
                                 (int(round(xh1)), int(round(yh1))),
                                 (0, 255, 255),
                                 2)
                        cv2.circle(vis,
                                   (int(round(xh1)), int(round(yh1))),
                                   6, (255, 0, 0), -1)  # blue

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
                            cv2.line(vis,
                                     (int(round(xh1)), int(round(yh1))),
                                     (int(round(x_target)), int(round(y_target))),
                                     (255, 0, 255),
                                     2)  # magenta
                            cv2.circle(vis,
                                       (int(round(x_target)), int(round(y_target))),
                                       6, (0, 0, 255), -1)  # red
                    else:
                        if t_direct is not None:
                            x_target = x0 + vx_hp * t_direct
                        else:
                            x_target = x0
                        cv2.line(vis,
                                 (xp, yp),
                                 (int(round(x_target)), int(round(y_target))),
                                 (0, 255, 255),
                                 2)  # yellow
                        cv2.circle(vis,
                                   (int(round(x_target)), int(round(y_target))),
                                   6, (0, 0, 255), -1)  # red

            else:
                # Single-puck logic
                mag = math.hypot(vx, vy)
                if (mag > VEL_THRESHOLD):
                    # Predict using puck velocity
                    x0, y0 = smoothed_puck
                    fb = compute_first_bounce(x0, y0, vx, vy, TABLE_W, TABLE_H)
                    if abs(vy) > 1e-3:
                        t_direct = (y_target - y0) / vy
                    else:
                        t_direct = None

                    need_bounce = False
                    if fb is not None and t_direct is not None:
                        t1, xh1, yh1, w1 = fb
                        if t1 > 0 and t1 < t_direct:
                            need_bounce = True

                    if need_bounce:
                        t1, xh1, yh1, w1 = fb
                        cv2.line(vis,
                                 (xp, yp),
                                 (int(round(xh1)), int(round(yh1))),
                                 (0, 255, 255),
                                 2)  # yellow
                        cv2.circle(vis,
                                   (int(round(xh1)), int(round(yh1))),
                                   6, (255, 0, 0), -1)  # blue

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
                            cv2.line(vis,
                                     (int(round(xh1)), int(round(yh1))),
                                     (int(round(x_target)), int(round(y_target))),
                                     (255, 0, 255),
                                     2)  # magenta
                            cv2.circle(vis,
                                       (int(round(x_target)), int(round(y_target))),
                                       6, (0, 0, 255), -1)  # red
                            time_until_impact = (t1 + t2) / FRAME_RATE
                    else:
                        if t_direct is not None and t_direct > 0:
                            x_target = x0 + vx * t_direct
                            cv2.line(vis,
                                     (xp, yp),
                                     (int(round(x_target)), int(round(y_target))),
                                     (0, 255, 255),
                                     2)  # yellow
                            cv2.circle(vis,
                                       (int(round(x_target)), int(round(y_target))),
                                       6, (0, 0, 255), -1)  # red
                            time_until_impact = t_direct / FRAME_RATE
                else:
                    # Not enough velocity for prediction
                    x_target = None
                    time_until_impact = None

        # Check if puck is in robot's half (top half) and update timer
        if puck_present and smoothed_puck:
            halfway_y = TABLE_H / 2.0
            current_time = time.time()
            
            # Track if puck crosses midline during follow-through
            if aggressive_mode_active and aggressive_phase == 3:
                if smoothed_puck[1] > halfway_y:
                    puck_crossed_midline = True
            
            # Check if puck is on robot's side (top half of table)
            if smoothed_puck[1] < halfway_y:
                if not puck_in_robot_half:
                    # Just entered robot's half
                    puck_in_robot_half = True
                    puck_in_robot_half_start_time = current_time
                    # Reset midline crossing flag when puck enters robot half
                    puck_crossed_midline = False
                elif not aggressive_mode_active and (current_time - puck_in_robot_half_start_time) > PUCK_IN_ROBOT_HALF_THRESHOLD:
                    # Puck has been in robot's half for over the threshold time, activate aggressive mode
                    aggressive_mode_active = True
                    aggressive_mode_start_time = current_time
                    aggressive_phase = 1  # Start with positioning phase
                    aggressive_phase_start_time = current_time
                    puck_crossed_midline = False
                    print("AGGRESSIVE MODE ACTIVATED - POSITIONING PHASE")
            else:
                # Puck is on human's side, reset the timer
                puck_in_robot_half = False
            
            # Check if we should transition between aggressive phases
            if aggressive_mode_active:
                # From positioning to striking
                if aggressive_phase == 1 and (current_time - aggressive_phase_start_time) > 1.0:
                    aggressive_phase = 2
                    aggressive_phase_start_time = current_time
                    print("AGGRESSIVE MODE - STRIKING PHASE")
                # From striking to follow-through
                elif aggressive_phase == 2 and (current_time - aggressive_phase_start_time) > 0.2:
                    aggressive_phase = 3
                    aggressive_phase_start_time = current_time
                    puck_crossed_midline = False
                    print("AGGRESSIVE MODE - FOLLOW-THROUGH PHASE (until puck crosses midline)")
                # From follow-through back to normal if puck crossed midline or timeout
                elif aggressive_phase == 3 and (puck_crossed_midline or 
                                             (current_time - aggressive_phase_start_time) > FOLLOW_THROUGH_TIMEOUT):
                    aggressive_mode_active = False
                    aggressive_phase = 0
                    if puck_crossed_midline:
                        print("Aggressive mode complete - puck crossed midline")
                    else:
                        print("Aggressive mode complete - follow-through timeout (10s)")
            
            # If aggressive mode is active, override the normal target with aggressive behavior
            if aggressive_mode_active and puck_present:
                # Define human's goal at the bottom center of table
                goal_x = TABLE_W / 2.0
                goal_y = TABLE_H  # Bottom of table
                
                # Calculate vector from puck to goal
                puck_x, puck_y = smoothed_puck
                goal_vector_x = goal_x - puck_x
                goal_vector_y = goal_y - puck_y
                
                # Normalize vector
                vector_length = math.hypot(goal_vector_x, goal_vector_y)
                if vector_length > 1e-3:
                    norm_vector_x = goal_vector_x / vector_length
                    norm_vector_y = goal_vector_y / vector_length
                    
                    # Phase 1: Position at intercept point along vector, at normal Y height
                    if aggressive_phase == 1:
                        # Find where the vector crosses the striker's y position
                        if abs(norm_vector_y) > 1e-3:  # Avoid division by zero
                            t = (y_target - puck_y) / norm_vector_y
                            x_target = puck_x + norm_vector_x * t
                            # Ensure x_target stays within bounds
                            x_target = max(0, min(x_target, TABLE_W))
                            time_until_impact = None  # Not hitting yet
                        else:
                            # Vector is horizontal, just position at puck's x
                            x_target = puck_x
                            time_until_impact = None
                    
                    # Phase 2: Strike position - go 10% past the puck
                    elif aggressive_phase == 2:
                        # Distance to travel along vector
                        extra_distance = TABLE_H * 0.1  # 10% of table height
                        
                        # Calculate strike point - we need to hit the puck and go 10% further
                        # First find where the vector crosses the striker's y position
                        if abs(norm_vector_y) > 1e-3:  # Avoid division by zero
                            t = (y_target - puck_y) / norm_vector_y
                            intercept_x = puck_x + norm_vector_x * t
                            
                            # Calculate additional distance vector component
                            dx = norm_vector_x * extra_distance
                            
                            # Set target beyond the intercept point
                            x_target = intercept_x + dx
                            # Ensure x_target stays within bounds
                            x_target = max(0, min(x_target, TABLE_W))
                            time_until_impact = 0.2  # Short time to trigger immediate movement
                            
                            # Store position for follow-through
                            last_strike_x_target = x_target
                        else:
                            # Vector is horizontal, strike from side with extra distance
                            x_target = puck_x + (1.0 if puck_x < TABLE_W/2.0 else -1.0) * extra_distance
                            time_until_impact = 0.2
                            last_strike_x_target = x_target
                    
                    # Phase 3: Follow through - stay in extended position
                    elif aggressive_phase == 3 and last_strike_x_target is not None:
                        # Hold the same strike position
                        x_target = last_strike_x_target
                        time_until_impact = None  # No need for impact timing in follow-through
                    
                    # Draw the aggressive target and vector on visualization
                    if aggressive_phase == 1:
                        mode_text = "POSITION"
                    elif aggressive_phase == 2:
                        mode_text = "STRIKE"
                    else:
                        mode_text = "FOLLOW"
                    
                    # Draw vector from puck to goal
                    cv2.line(vis,
                            (int(round(puck_x)), int(round(puck_y))),
                            (int(round(goal_x)), int(round(goal_y))),
                            (255, 0, 255),  # Magenta for goal vector
                            1)  # Thin line
                    
                    # Draw target position
                    cv2.line(vis,
                            (int(round(puck_x)), int(round(puck_y))),
                            (int(round(x_target)), int(round(y_target))),
                            (0, 0, 255),  # Red line for aggressive mode
                            3)  # Thicker line
                    cv2.circle(vis,
                            (int(round(x_target)), int(round(y_target))),
                            8, (0, 0, 255), -1)  # Red, larger dot
                    
                    # Override hit mode - explicitly NOT using hit mode for aggressive behavior
                    hit_mode_active = False

        # FPS counter update
        fps_count += 1
        now = time.time()
        elapsed = now - fps_start
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count = 0
            fps_start = now
            
        # Send command over serial on every frame if we have a valid target
        if x_target is not None and ser is not None:
            try:
                # Determine if hit mode should be activated
                hit_mode_trigger = (time_until_impact is not None and time_until_impact < 0.4) or \
                            (puck_present and smoothed_puck and abs(smoothed_puck[1] - y_target) < TABLE_H * 0.15)
                
                current_time = time.time()
                
                # If hit mode triggered, start/extend the timer
                if hit_mode_trigger:
                    hit_mode_active = True
                    hit_mode_start_time = current_time
                
                # Check if we should exit hit mode
                if hit_mode_active and (current_time - hit_mode_start_time) > HIT_MODE_DURATION:
                    hit_mode_active = False
                
                # Use hit mode state for position adjustment
                in_hit_mode = hit_mode_active
                
                # Apply adjustment logic
                table_width = float(TABLE_W)
                adjusted_x_target = x_target
                
                y_target_adder = 0.0
                if in_hit_mode:
                    y_target_adder = 0.05 * TABLE_H

                # Percent X and Y
                percent_x = x_target / TABLE_W
                percent_y = (y_target + y_target_adder) / TABLE_H

                # min and max values
                percent_x = max(percent_x, 0.0)
                percent_x = min(percent_x, 1.0)
                percent_y = max(percent_y, 0.0)
                percent_y = min(percent_y, 1.0)
   
                # Scale coordinates
                scaled_x = int((percent_x) * 2857)
                scaled_y = int((percent_y) * 4873)             
                
                # Format and send command
                msg = f"M{scaled_x:04d}{scaled_y:04d}\r\n"
                
                # Send each byte individually with a small delay between them
                for byte in msg.encode('ascii'):
                    ser.write(bytes([byte]))  # Send single byte
                    time.sleep(0.001)  # 1ms delay between bytes
            except Exception as e:
                print(f"Error sending command: {e}")
            
        # Print status to terminal once per second when we have a valid target
        current_time = time.time()
        if x_target is not None and (not hasattr(main_loop, "last_print_time") or (current_time - main_loop.last_print_time) >= 1.0):
            # Use the persistent hit mode state for status display
            mode_str = "HIT" if hit_mode_active else "PREDICT"
            status_msg = f"{mode_str}: Target={x_target:.1f},{y_target:.1f}"
            if time_until_impact is not None:
                status_msg += f" Time={time_until_impact:.2f}s"
            else:
                status_msg += " Time=unknown"
                
            # Add serial command details if available
            if ser is not None:
                status_msg += f" Command=M{scaled_x:04d}{scaled_y:04d}"
                
            print(status_msg)
            main_loop.last_print_time = current_time

        # Overlay FPS
        cv2.putText(vis,
                    f"FPS: {fps_display:.1f}",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)
        
        # Add mode indicator text
        if aggressive_mode_active:
            if aggressive_phase == 1:
                mode_text = "Aggressive-Position"
                mode_color = (255, 0, 255)  # Magenta for positioning
            elif aggressive_phase == 2:
                mode_text = "Aggressive-Strike"
                mode_color = (0, 0, 255)  # Red for striking
            else:
                mode_text = "Aggressive-Follow"
                mode_color = (255, 165, 0)  # Orange for follow-through
        elif hit_mode_active:
            mode_text = "Hit"
            mode_color = (0, 0, 255)  # Red for Hit
        else:
            mode_text = "Predict"
            mode_color = (0, 255, 0)  # Green for Predict
            
        cv2.putText(vis,
                    mode_text,
                    (30, 70),  # Position under FPS counter
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    mode_color,
                    2)

        # Display in a regular window with more reasonable size
        vis_display = cv2.resize(vis, (800, 600), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win, vis_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()
        print("\nSerial port closed.\n")

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air Hockey Table Detection on Raspberry Pi")
    parser.add_argument(
        "--mode",
        choices=["calibrate_frame", "calibrate_hsv", "run"],
        required=True,
        help="Mode = calibrate_frame | calibrate_hsv | run"
    )
    args = parser.parse_args()

    if args.mode == "calibrate_frame":
        calibrate_frame()
    elif args.mode == "calibrate_hsv":
        calibrate_hsv()
    elif args.mode == "run":
        main_loop()
    else:
        print("Unknown mode. Use --mode calibrate_frame / calibrate_hsv / run.")
