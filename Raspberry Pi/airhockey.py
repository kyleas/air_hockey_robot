#!/usr/bin/env python3
# airhockey.py
#
# Complete Python script with:
#  1) Frame calibration (windowed, not fullscreen)
#  2) HSV calibration (windowed; live raw + masked preview)
#  3) Main detection loop (fullscreen; dual‐circle logic vs. single‐puck logic)
#  4) Exponential smoothing on puck to reduce jitter
#  5) Drawing:
#       • Two circles: handle→puck vector, then puck→20% (direct or reflection)
#       • Single puck, low velocity & puck below halfway: vertical line
#       • Single puck, else: puck→20% (direct or reflection)
#  6) Serial output: “x_target,time_until_impact\n” (only for single‐puck velocity cases)
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

# Serial port to STM32 (adjust if needed; e.g., '/dev/ttyACM0' or '/dev/ttyUSB0')
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200

# Minimum circle radius (in warped pixels) to accept as “red circle”
MIN_RADIUS = 15

# Exponential smoothing factor (α) for puck position (0 < α < 1). Larger α = less smoothing.
SMOOTHING_ALPHA = 0.3

# Velocity threshold (pixels/frame) for distinguishing “fast” vs “slow”
VEL_THRESHOLD = 2.0

# How many clicks per side during frame calibration
CLICKS_PER_SIDE = 2

# Tolerances to expand from sampled HSV values
H_MARGIN = 10    # hue ±10 (0–180)
S_MARGIN = 40    # sat ±40 (0–255)
V_MARGIN = 40    # val ±40 (0–255)

# Frame rate (fps) assumption for time‐to‐impact calculation
FRAME_RATE = 30.0

# ------------------------------------------------------------------------------
# GLOBALS FOR MOUSE CALLBACKS & SMOOTHING STATE
# ------------------------------------------------------------------------------
clicks        = []      # raw pixel coords during frame calibration
hsv_samples   = []      # list of (h,s,v) tuples during HSV calibration

# Smoothed puck positions
smoothed_puck       = None   # (x, y)
prev_smoothed_puck  = None   # (x, y)

# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------

def line_from_two_points(pt1, pt2):
    """
    Returns line coefficients (a, b, c) for line equation: a*x + b*y + c = 0
    passing through pt1 and pt2.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    a = float(y1 - y2)
    b = float(x2 - x1)
    c = float(x1 * y2 - x2 * y1)
    return (a, b, c)

def intersect_lines(l1, l2):
    """
    Given l1=(a1,b1,c1) and l2=(a2,b2,c2), solve intersection:
      a1*x + b1*y + c1 = 0
      a2*x + b2*y + c2 = 0
    Returns (x, y). If nearly parallel, returns (0.0, 0.0).
    """
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    denom = (a1 * b2 - a2 * b1)
    if abs(denom) < 1e-8:
        return (0.0, 0.0)
    x = (b1 * c2 - b2 * c1) / denom
    y = (c1 * a2 - c2 * a1) / denom
    return (x, y)

def save_warp_matrix(filename, matrix, width, height):
    """
    Store the warp matrix (3×3), width, height into a JSON file.
    """
    data = {
        "matrix": matrix.tolist(),
        "width": float(width),
        "height": float(height)
    }
    with open(filename, "w") as f:
        json.dump(data, f)

def load_warp_matrix(filename):
    """
    Load warp matrix (np.float32 3×3), width, height from JSON.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    mat = np.array(data["matrix"], dtype=np.float32)
    w   = int(data["width"])
    h   = int(data["height"])
    return mat, w, h

def save_hsv_ranges(filename, hsv_dict):
    """
    hsv_dict = { "h_min":…, "h_max":…, "s_min":…, "s_max":…, "v_min":…, "v_max":… }
    """
    with open(filename, "w") as f:
        json.dump(hsv_dict, f)

def load_hsv_ranges(filename):
    """
    Returns: (lower_bound, upper_bound) as np.array([H,S,V], dtype=np.uint8).
    """
    with open(filename, "r") as f:
        data = json.load(f)
    lower = np.array([data["h_min"], data["s_min"], data["v_min"]], dtype=np.uint8)
    upper = np.array([data["h_max"], data["s_max"], data["v_max"]], dtype=np.uint8)
    return lower, upper

def mouse_callback_frame(event, x, y, flags, param):
    """
    Callback for collecting clicks during frame calibration.
    """
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))

# ------------------------------------------------------------------------------
# CALIBRATION: FRAME WARP
# ------------------------------------------------------------------------------

def calibrate_frame():
    """
    Pops up an 800×600 window with live camera feed.
    User clicks:
      • 2 points on TOP edge → press 'n'
      • 2 points on RIGHT edge → press 'n'
      • 2 points on BOTTOM edge → press 'n'
      • 2 points on LEFT edge → press 'n'
    Computes perspective warp from those 8 points and saves to warp_matrix.json.
    """
    global clicks
    clicks = []
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Ensure the Pi camera is enabled.")
        return

    window_name = "Frame Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.setMouseCallback(window_name, mouse_callback_frame)

    side_names = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    side_idx = 0

    print("== FRAME CALIBRATION ==")
    print("Click 2 points on each side of the table frame in order:")
    print("  1) TOP edge → press 'n'")
    print("  2) RIGHT edge → press 'n'")
    print("  3) BOTTOM edge → press 'n'")
    print("  4) LEFT edge → press 'n'")
    print("Press 'q' to abort at any time.")

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
                print(f"  >> Must click {CLICKS_PER_SIDE} points on {side_names[side_idx]} first.")
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
        print(f"ERROR: Expected 8 points, got {len(clicks)}. Aborting.")
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
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    save_warp_matrix(FRAME_CALIB_FILE, M, maxWidth, maxHeight)
    print(f"\n[OK] Saved warp matrix to '{FRAME_CALIB_FILE}' ({maxWidth}×{maxHeight}).\n")

# ------------------------------------------------------------------------------
# CALIBRATION: HSV RANGE (WINDOWED)
# ------------------------------------------------------------------------------

def calibrate_hsv():
    """
    Opens two 800×600 windows:
      • "HSV Calibration - Raw": live BGR feed for clicking
      • "HSV Calibration - Masked": live masked preview
    Click on each visible red circle in RAW. MASKED updates in real time.
    Press 'q' when done; saves final HSV bounds to hsv_ranges.json.
    """
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
            h_min = max(0, min(hs) - H_MARGIN)
            h_max = min(180, max(hs) + H_MARGIN)
            s_min = max(0, min(ss) - S_MARGIN)
            s_max = min(255, max(ss) + S_MARGIN)
            v_min = max(0, min(vs) - V_MARGIN)
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
    h_min = max(0, min(hs) - H_MARGIN)
    h_max = min(180, max(hs) + H_MARGIN)
    s_min = max(0, min(ss) - S_MARGIN)
    s_max = min(255, max(ss) + S_MARGIN)
    v_min = max(0, min(vs) - V_MARGIN)
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
    """
    Reflect (vx, vy) across the specified wall.
    """
    if hit_wall in ("left", "right"):
        return (-vx, vy)
    elif hit_wall in ("top", "bottom"):
        return (vx, -vy)
    else:
        return (vx, vy)

def compute_first_bounce(x0, y0, vx, vy, W, H):
    """
    Given ray from (x0,y0) in direction (vx,vy),
    compute first intersection with table walls.
    Walls: x=0 'left', x=W 'right', y=0 'top', y=H 'bottom'.
    Returns (t_hit, x_hit, y_hit, wall_name) or None.
    """
    candidates = []
    # LEFT
    if vx < 0:
        t = (0 - x0) / vx
        if t > 1e-6:
            y_hit = y0 + t * vy
            if 0 <= y_hit <= H:
                candidates.append((t, 0.0, y_hit, "left"))
    # RIGHT
    if vx > 0:
        t = (W - x0) / vx
        if t > 1e-6:
            y_hit = y0 + t * vy
            if 0 <= y_hit <= H:
                candidates.append((t, float(W), y_hit, "right"))
    # TOP
    if vy < 0:
        t = (0 - y0) / vy
        if t > 1e-6:
            x_hit = x0 + t * vx
            if 0 <= x_hit <= W:
                candidates.append((t, x_hit, 0.0, "top"))
    # BOTTOM
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
    """
    Runs detection:
      • Warp each frame
      • Threshold red, find HoughCircles
      • If 2 circles: handle vs puck → draw handle→puck → draw puck→20% (direct or reflection)
      • If 1 circle: compute velocity; if slow & below halfway, vertical; else draw puck→20%
      • Send x_target,time_until_impact (only for single-puck velocity cases)
    """
    global smoothed_puck, prev_smoothed_puck

    if not os.path.exists(FRAME_CALIB_FILE):
        print(f"ERROR: '{FRAME_CALIB_FILE}' missing. Run --mode calibrate_frame.")
        return
    if not os.path.exists(HSV_CALIB_FILE):
        print(f"ERROR: '{HSV_CALIB_FILE}' missing. Run --mode calibrate_hsv.")
        return

    warp_matrix, TABLE_W, TABLE_H = load_warp_matrix(FRAME_CALIB_FILE)
    hsv_lower, hsv_upper         = load_hsv_ranges(HSV_CALIB_FILE)

    # y coordinate at 20% from top
    y_target = 0.2 * TABLE_H

    # Open serial if possible
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
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n== RUNNING DETECTION: press 'q' to quit ==\n")

    smoothed_puck = None
    prev_smoothed_puck = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        warped = cv2.warpPerspective(frame, warp_matrix, (TABLE_W, TABLE_H))
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        masked_color = cv2.bitwise_and(warped, warped, mask=mask)
        gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9,9), 2)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=MIN_RADIUS*2,
            param1=50,
            param2=30,
            minRadius=MIN_RADIUS,
            maxRadius=0
        )

        vis = warped.copy()

        handle_present = False
        puck_present   = False
        x_target = None
        time_until_impact = None

        if circles is not None and len(circles[0]) >= 1:
            detected = np.round(circles[0, :]).astype("int")
            detected = [c for c in detected if c[2] >= MIN_RADIUS]
            if len(detected) >= 1:
                if len(detected) >= 2:
                    pair = sorted(detected, key=lambda x: x[2], reverse=True)[:2]
                    c0, c1 = pair[0], pair[1]
                    # The one with smaller y is puck (higher on table)
                    if c0[1] < c1[1]:
                        puck_raw   = (float(c0[0]), float(c0[1]))
                        handle_raw = (float(c1[0]), float(c1[1]))
                        r_puck     = c0[2]
                        r_handle   = c1[2]
                    else:
                        puck_raw   = (float(c1[0]), float(c1[1]))
                        handle_raw = (float(c0[0]), float(c0[1]))
                        r_puck     = c1[2]
                        r_handle   = c0[2]
                    handle_present = True
                else:
                    c = detected[0]
                    handle_raw = None
                    puck_raw   = (float(c[0]), float(c[1]))
                    r_puck     = c[2]

                # Smooth puck
                if smoothed_puck is None:
                    smoothed_puck = puck_raw
                else:
                    smoothed_puck = (
                        SMOOTHING_ALPHA * puck_raw[0] + (1 - SMOOTHING_ALPHA) * smoothed_puck[0],
                        SMOOTHING_ALPHA * puck_raw[1] + (1 - SMOOTHING_ALPHA) * smoothed_puck[1]
                    )
                puck_present = True

                # Compute velocity
                if prev_smoothed_puck is not None:
                    vx = smoothed_puck[0] - prev_smoothed_puck[0]
                    vy = smoothed_puck[1] - prev_smoothed_puck[1]
                else:
                    vx, vy = 0.0, 0.0
                prev_smoothed_puck = smoothed_puck

                xp, yp = int(round(smoothed_puck[0])), int(round(smoothed_puck[1]))
                cv2.circle(vis, (xp, yp), r_puck, (255, 255, 0), 2)  # puck = cyan
                cv2.circle(vis, (xp, yp), 4, (0, 255, 255), -1)

                if handle_present:
                    xh, yh = int(round(handle_raw[0])), int(round(handle_raw[1]))
                    cv2.circle(vis, (xh, yh), r_handle, (0, 255, 0), 2)  # handle = green
                    cv2.circle(vis, (xh, yh), 4, (0, 0, 255), -1)

                    # Draw handle → puck
                    cv2.line(vis,
                             (xh, yh),
                             (xp, yp),
                             (0, 255, 255),
                             2)  # yellow

                    # Now draw puck → 20% using handle→puck vector
                    x0, y0 = smoothed_puck
                    vx_hp = x0 - handle_raw[0]
                    vy_hp = y0 - handle_raw[1]

                    # Direct path parameters: from puck along (vx_hp, vy_hp)
                    fb = compute_first_bounce(x0, y0, vx_hp, vy_hp, TABLE_W, TABLE_H)

                    # Time (in frames) for direct to reach y_target:
                    # Solve y0 + t * vy_hp = y_target → t_d = (y_target - y0)/vy_hp
                    if abs(vy_hp) > 1e-3:
                        t_direct = (y_target - y0) / vy_hp
                    else:
                        t_direct = None

                    # Decide if bounce is needed before target
                    need_bounce = False
                    if fb is not None and t_direct is not None:
                        t1, xh1, yh1, w1 = fb
                        if t1 > 0 and t1 < t_direct:
                            need_bounce = True

                    if need_bounce:
                        # Draw puck → first bounce in yellow
                        cv2.line(vis,
                                 (xp, yp),
                                 (int(round(xh1)), int(round(yh1))),
                                 (0, 255, 255),
                                 2)
                        cv2.circle(vis,
                                   (int(round(xh1)), int(round(yh1))),
                                   6, (255, 0, 0), -1)  # blue

                        # Reflect and draw from bounce → target
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
                        # Direct path: draw puck → target
                        x_target = x0 + ((t_direct if t_direct else 0) * vx_hp) if t_direct else x0
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
                    # Vertical if low velocity AND puck below halfway (yp > H/2)
                    if (mag < VEL_THRESHOLD) and (yp > (TABLE_H / 2.0)):
                        cv2.line(vis,
                                 (xp, yp),
                                 (xp, int(round(y_target))),
                                 (0, 255, 255),
                                 2)  # yellow
                        cv2.circle(vis,
                                   (xp, int(round(y_target))),
                                   6, (0, 0, 255), -1)  # red
                        x_target = float(xp)
                        if vy < -1e-3:
                            t_frames = (smoothed_puck[1] - y_target) / (-vy)
                            time_until_impact = t_frames / FRAME_RATE
                    else:
                        # Predict using puck velocity vector
                        x0, y0 = smoothed_puck
                        # Direct: (vx, vy)
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
                                # Compute time
                                time_until_impact = (t1 + t2) / FRAME_RATE
                        else:
                            # Direct
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

                # Send serial only for single-puck velocity cases
                if handle_present is False and (x_target is not None) and (time_until_impact is not None):
                    try:
                        msg = f"{x_target:.2f},{time_until_impact:.2f}\n"
                        if ser is not None:
                            ser.write(msg.encode('ascii'))
                    except:
                        pass

        # Display fullscreen (1080×1440)
        vis_display = cv2.resize(vis, (1080, 1440), interpolation=cv2.INTER_LINEAR)
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
