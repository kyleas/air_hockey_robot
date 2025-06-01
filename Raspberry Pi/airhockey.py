#!/usr/bin/env python3
# airhockey.py
#
# Updated to:
#  1) Smooth circle positions with an exponential moving average (EMA) so detection is less jittery.
#  2) HSV‐calibration now shows TWO windows: the raw video (for clicks) and the real-time "mask" of what's being filtered.
#  3) When two red circles are found, the one with the larger y‐coordinate is considered the "handle" (further back). We draw the ray:
#       handle → puck → first‐wall intersection → second‐wall intersection.
#  4) All OpenCV windows are opened in FULLSCREEN/WINDOW_NORMAL so that they fill the Pi’s screen.
#
# To install dependencies on Raspbian:
#    sudo apt update
#    sudo apt install python3-pip
#    pip3 install opencv-python numpy pyserial
#
# Usage:
#    python3 airhockey.py --mode calibrate_frame
#    python3 airhockey.py --mode calibrate_hsv
#    python3 airhockey.py --mode run
#
# Serial protocol: sends lines of ASCII: "x_handle,y_handle,vx_ref,vy_ref\n"
#    where (x_handle,y_handle) is the smoothed handle position, and (vx_ref,vy_ref) is the reflected vector
#    after the first bounce.
#
# To autostart on boot, create a systemd service that runs:
#    ExecStart=/usr/bin/python3 /home/pi/airhockey.py --mode run
# and enable it (see bottom of file).
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

# Serial port to STM32 (adjust if needed; e.g. '/dev/ttyACM0' or '/dev/ttyUSB0')
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200

# Minimum circle radius (in warped pixels) to accept as “red circle”
MIN_RADIUS = 15

# Exponential smoothing factor (α) for positions (0 < α < 1). Higher = less smoothing,
# Lower = more smoothing. 0.3 is a good compromise.
SMOOTHING_ALPHA = 0.3

# How many clicks per side during frame calibration
CLICKS_PER_SIDE = 2

# Tolerances to expand from sampled HSV values
H_MARGIN = 10    # hue ±10 (0–180)
S_MARGIN = 40    # sat ±40 (0–255)
V_MARGIN = 40    # val ±40 (0–255)

# ------------------------------------------------------------------------------
# GLOBALS FOR MOUSE CALLBACKS & SMOOTHING STATE
# ------------------------------------------------------------------------------
clicks     = []      # raw pixel coords during frame calibration
hsv_samples = []     # list of (h,s,v) tuples during HSV calibration

# Smoothed positions (None until first valid detection)
smoothed_handle = None   # (x_h, y_h)
smoothed_puck   = None   # (x_p, y_p)

# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------

def line_from_two_points(pt1, pt2):
    """
    Returns line coefficients (a, b, c) for line equation: a*x + b*y + c = 0
    passing through pt1=(x1,y1) and pt2=(x2,y2).
    """
    x1, y1 = pt1
    x2, y2 = pt2
    a = float(y1 - y2)
    b = float(x2 - x1)
    c = float(x1 * y2 - x2 * y1)
    return (a, b, c)

def intersect_lines(l1, l2):
    """
    Given l1=(a1,b1,c1) and l2=(a2,b2,c2), solve for their intersection:
        a1*x + b1*y + c1 = 0
        a2*x + b2*y + c2 = 0
    Returns (x, y) as floats. If nearly parallel, returns (0.0, 0.0).
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
    Store the warp matrix (3x3), width, height into a JSON file.
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
    Load warp matrix (np.float32 3x3), width, height from JSON.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    mat = np.array(data["matrix"], dtype=np.float32)
    w   = int(data["width"])
    h   = int(data["height"])
    return mat, w, h

def save_hsv_ranges(filename, hsv_dict):
    """
    hsv_dict = { "h_min":..., "h_max":..., "s_min":..., "s_max":..., "v_min":..., "v_max":... }
    """
    with open(filename, "w") as f:
        json.dump(hsv_dict, f)

def load_hsv_ranges(filename):
    """
    Returns: (lower_bound, upper_bound) each as np.array([H,S,V], dtype=np.uint8)
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

def mouse_callback_hsv(event, x, y, flags, param):
    """
    Callback for collecting HSV samples: when user clicks, record HSV pixel at that location.
    """
    global hsv_samples
    frame_hsv = param["hsv_frame"]
    if event == cv2.EVENT_LBUTTONDOWN:
        h, s, v = frame_hsv[y, x]
        hsv_samples.append((int(h), int(s), int(v)))
        print(f"  [HSV SAMPLE] at ({x},{y}):  H={h}, S={s}, V={v}")

# ------------------------------------------------------------------------------
# CALIBRATION: FRAME WARP
# ------------------------------------------------------------------------------

def calibrate_frame():
    """
    Pops up a window showing a live camera feed. Instruct the user to click:
      - 2 points on the TOP edge, then press 'n'
      - 2 points on the RIGHT edge, then press 'n'
      - 2 points on the BOTTOM edge, then press 'n'
      - 2 points on the LEFT edge, then press 'n'
    Once 8 points are collected, compute lines, intersections, perspective warp, and save to JSON.
    """
    global clicks
    clicks = []
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Make sure the Pi camera is enabled and /dev/video0 is available.")
        return

    window_name = "Frame Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback_frame)

    side_names = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    side_idx = 0

    print("== FRAME CALIBRATION ==")
    print("You will click 2 points on each side of the table frame, in order:")
    print("  1) TOP edge: click 2 points → press 'n'")
    print("  2) RIGHT edge: click 2 points → press 'n'")
    print("  3) BOTTOM edge: click 2 points → press 'n'")
    print("  4) LEFT edge: click 2 points → press 'n'")
    print("Press 'q' at any time to abort.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw existing clicks
        vis = frame.copy()
        for (x, y) in clicks:
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)

        cv2.putText(vis, f"Click 2 points on the {side_names[side_idx]} edge", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('n'):
            # Ensure exactly 2 new clicks were added for this side
            if len(clicks) < (side_idx+1)*CLICKS_PER_SIDE:
                print(f"  >> You must click exactly {CLICKS_PER_SIDE} points on the {side_names[side_idx]} edge first.")
                continue
            side_idx += 1
            if side_idx >= 4:
                # Collected 8 total clicks
                break
            print(f"Now click 2 points on the {side_names[side_idx]} edge, then press 'n'.")
        elif key == ord('q'):
            print("Calibration aborted by user.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(clicks) != 8:
        print(f"ERROR: Expected 8 clicks (2/sides). Got {len(clicks)}. Aborting.")
        return

    # Group clicks by side: 0-1=top, 2-3=right, 4-5=bottom, 6-7=left
    top_pts    = clicks[0:2]
    right_pts  = clicks[2:4]
    bottom_pts = clicks[4:6]
    left_pts   = clicks[6:8]

    # Compute line equations
    l_top    = line_from_two_points(top_pts[0],    top_pts[1])
    l_right  = line_from_two_points(right_pts[0],  right_pts[1])
    l_bottom = line_from_two_points(bottom_pts[0], bottom_pts[1])
    l_left   = line_from_two_points(left_pts[0],   left_pts[1])

    # Intersect lines to get corners
    tl = intersect_lines(l_top,    l_left)
    tr = intersect_lines(l_top,    l_right)
    br = intersect_lines(l_bottom, l_right)
    bl = intersect_lines(l_bottom, l_left)

    # Compute max width/height in pixels
    widthA  = math.hypot(br[0]-bl[0], br[1]-bl[1])
    widthB  = math.hypot(tr[0]-tl[0], tr[1]-tl[1])
    maxWidth  = max(int(widthA), int(widthB))

    heightA = math.hypot(tr[0]-br[0], tr[1]-br[1])
    heightB = math.hypot(tl[0]-bl[0], tl[1]-bl[1])
    maxHeight = max(int(heightA), int(heightB))

    # Build source/dest arrays for getPerspectiveTransform
    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([[0,0],
                        [maxWidth-1, 0],
                        [maxWidth-1, maxHeight-1],
                        [0, maxHeight-1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    save_warp_matrix(FRAME_CALIB_FILE, M, maxWidth, maxHeight)
    print(f"\n[OK] Frame calibration saved to '{FRAME_CALIB_FILE}'.")
    print(f"     Warped table size = {maxWidth} × {maxHeight} px.\n")

# ------------------------------------------------------------------------------
# CALIBRATION: HSV RANGE
# ------------------------------------------------------------------------------

def calibrate_hsv():
    """
    Captures a single frame from camera, then lets the user click on all visible red circles.
    Two windows appear:
      1) "HSV Calibration - Raw"    → for clicking
      2) "HSV Calibration - Masked" → shows what is currently being masked by the HSV range.
    Press 'q' when done.
    """
    global hsv_samples
    hsv_samples = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera for HSV calibration.")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Could not grab a frame for HSV calibration.")
        return

    # Prepare windows
    win_raw    = "HSV Calibration - Raw"
    win_masked = "HSV Calibration - Masked"
    cv2.namedWindow(win_raw,    cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_masked, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_raw,    cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(win_masked, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Convert once for sampling
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.setMouseCallback(win_raw, mouse_callback_hsv, {"hsv_frame": frame_hsv})

    print("\n== HSV CALIBRATION ==")
    print("Click on every red circle in the RAW window.")
    print("As you click, the MASKED window updates to show what pixels are currently being "
          "matched by the HSV range (based on your samples).")
    print("When done, press 'q' (in either window) to finish.\n")

    while True:
        # Compute current HSV min/max ± margins from all samples so far
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
            # If no samples yet, use a dummy range that matches nothing
            h_min = 0;   h_max = 0
            s_min = 0;   s_max = 0
            v_min = 0;   v_max = 0

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        # Show the RAW frame with instruction overlay
        vis_raw = frame.copy()
        cv2.putText(vis_raw, f"Clicks={len(hsv_samples)}; Press 'q' when done.", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
        cv2.imshow(win_raw, vis_raw)

        # Show the MASKED result in real-time
        mask = cv2.inRange(frame_hsv, lower, upper)
        masked_vis = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.putText(masked_vis, f"H=({h_min}-{h_max}), S=({s_min}-{s_max}), V=({v_min}-{v_max})",
                    (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow(win_masked, masked_vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if not hsv_samples:
        print("No HSV samples were collected; aborting.")
        return

    hsv_dict = {
        "h_min": int(h_min),
        "h_max": int(h_max),
        "s_min": int(s_min),
        "s_max": int(s_max),
        "v_min": int(v_min),
        "v_max": int(v_max)
    }
    save_hsv_ranges(HSV_CALIB_FILE, hsv_dict)

    print(f"\n[OK] HSV calibration saved to '{HSV_CALIB_FILE}':")
    print(f"     H ∈ [{h_min}, {h_max}], S ∈ [{s_min}, {s_max}], V ∈ [{v_min}, {v_max}]\n")

# ------------------------------------------------------------------------------
# BOUNCE / REFLECTION LOGIC
# ------------------------------------------------------------------------------

def reflect_vector(vx, vy, hit_wall):
    """
    Reflect (vx, vy) across the wall. 
    hit_wall is one of 'left','right','top','bottom'.
    Returns (vx_ref, vy_ref).
    """
    if hit_wall in ("left", "right"):
        return (-vx, vy)
    elif hit_wall in ("top", "bottom"):
        return (vx, -vy)
    else:
        return (vx, vy)

def compute_first_bounce(x0, y0, vx, vy, W, H):
    """
    Given a ray starting at (x0,y0) with direction (vx,vy), compute which wall it hits first.
    Walls: x=0 ('left'), x=W ('right'), y=0 ('top'), y=H ('bottom').
    Returns (t_hit, x_hit, y_hit, wall_name) where t_hit > 0.
    If no positive intersection remains, returns None.
    """
    candidates = []
    # LEFT wall (x=0) if vx<0
    if vx < 0:
        t = (0 - x0) / vx
        if t > 1e-6:  # positive and not at t≈0
            y_hit = y0 + t*vy
            if 0 <= y_hit <= H:
                candidates.append((t, 0.0, y_hit, "left"))
    # RIGHT wall (x=W) if vx>0
    if vx > 0:
        t = (W - x0) / vx
        if t > 1e-6:
            y_hit = y0 + t*vy
            if 0 <= y_hit <= H:
                candidates.append((t, float(W), y_hit, "right"))
    # TOP wall (y=0) if vy<0
    if vy < 0:
        t = (0 - y0) / vy
        if t > 1e-6:
            x_hit = x0 + t*vx
            if 0 <= x_hit <= W:
                candidates.append((t, x_hit, 0.0, "top"))
    # BOTTOM wall (y=H) if vy>0
    if vy > 0:
        t = (H - y0) / vy
        if t > 1e-6:
            x_hit = x0 + t*vx
            if 0 <= x_hit <= W:
                candidates.append((t, x_hit, float(H), "bottom"))

    if not candidates:
        return None

    # pick the smallest positive t
    t_hit, xh, yh, wall = min(candidates, key=lambda e: e[0])
    return (t_hit, xh, yh, wall)

# ------------------------------------------------------------------------------
# MAIN DETECTION + SERIAL SENDING (with smoothing)
# ------------------------------------------------------------------------------

def main_loop():
    """
    Loads warp matrix + HSV thresholds, opens serial port, and runs main detection loop:
    - Grab frame, warp, threshold for red, HoughCircles
    - Identify "handle" (circle with larger y) & "puck" (other circle)
    - Smooth both positions via EMA
    - Compute ray: handle → puck → first‐wall bounce → second‐wall bounce
    - Draw everything, send (handle_x, handle_y, vx_ref, vy_ref) via serial
    """
    global smoothed_handle, smoothed_puck

    # Ensure calibration files exist
    if not os.path.exists(FRAME_CALIB_FILE):
        print(f"ERROR: Frame calibration file '{FRAME_CALIB_FILE}' not found. Run --mode calibrate_frame first.")
        return
    if not os.path.exists(HSV_CALIB_FILE):
        print(f"ERROR: HSV calibration file '{HSV_CALIB_FILE}' not found. Run --mode calibrate_hsv first.")
        return

    warp_matrix, TABLE_W, TABLE_H = load_warp_matrix(FRAME_CALIB_FILE)
    hsv_lower, hsv_upper       = load_hsv_ranges(HSV_CALIB_FILE)

    # Open serial to STM32 (if available)
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2.0)  # give STM32 time to reset if needed
        print(f"[OK] Opened serial on {SERIAL_PORT} @ {BAUD_RATE} baud.")
    except Exception as e:
        print(f"[WARN] Could not open serial port '{SERIAL_PORT}': {e}")
        ser = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera for main loop.")
        return

    win = "AirHockey Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n== RUNNING DETECTION (press 'q' to quit) ==\n")

    # Reset any previous smoothing state
    smoothed_handle = None
    smoothed_puck   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1) Warp to table view
        warped = cv2.warpPerspective(frame, warp_matrix, (TABLE_W, TABLE_H))

        # 2) Convert to HSV, threshold
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        # 3) Blur & HoughCircles
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

        if circles is not None and len(circles[0]) >= 2:
            circles = np.round(circles[0, :]).astype("int")
            # Filter out anything that is obviously too small (just in case)
            circles = [c for c in circles if c[2] >= MIN_RADIUS]
            if len(circles) >= 2:
                # We only care about the two “most likely” circles. User-supplied logic:
                #   → The handle is the circle with the larger y-coordinate (further back).
                #   → The other circle (lower y) is treated as the puck.
                # If >2 circles detected, pick the two whose combined radii are largest, then re-order by y.
                circles = sorted(circles, key=lambda c: c[2], reverse=True)[:2]
                # Now identify handle vs puck by y:
                c0, c1 = circles[0], circles[1]
                # c = (x, y, r)
                if c0[1] > c1[1]:
                    handle_raw = (float(c0[0]), float(c0[1]))
                    puck_raw   = (float(c1[0]), float(c1[1]))
                    r_handle   = c0[2]
                    r_puck     = c1[2]
                else:
                    handle_raw = (float(c1[0]), float(c1[1]))
                    puck_raw   = (float(c0[0]), float(c0[1]))
                    r_handle   = c1[2]
                    r_puck     = c0[2]

                # Exponential smoothing (EMA) on both positions
                if smoothed_handle is None:
                    smoothed_handle = handle_raw
                else:
                    smoothed_handle = (
                        SMOOTHING_ALPHA * handle_raw[0] + (1 - SMOOTHING_ALPHA)*smoothed_handle[0],
                        SMOOTHING_ALPHA * handle_raw[1] + (1 - SMOOTHING_ALPHA)*smoothed_handle[1]
                    )
                if smoothed_puck is None:
                    smoothed_puck = puck_raw
                else:
                    smoothed_puck = (
                        SMOOTHING_ALPHA * puck_raw[0] + (1 - SMOOTHING_ALPHA)*smoothed_puck[0],
                        SMOOTHING_ALPHA * puck_raw[1] + (1 - SMOOTHING_ALPHA)*smoothed_puck[1]
                    )

                # Draw the smoothed handle & puck circles
                xh, yh = int(round(smoothed_handle[0])), int(round(smoothed_handle[1]))
                xp, yp = int(round(smoothed_puck[0])),   int(round(smoothed_puck[1]))
                cv2.circle(vis, (xh, yh), r_handle, (0, 255, 0), 2)   # handle in green
                cv2.circle(vis, (xh, yh), 4, (0, 0, 255), -1)
                cv2.circle(vis, (xp, yp), r_puck,   (255, 255, 0), 2) # puck in cyan
                cv2.circle(vis, (xp, yp), 4, (0, 255, 255), -1)

                # 4) Compute vector from handle→puck
                vx = smoothed_puck[0] - smoothed_handle[0]
                vy = smoothed_puck[1] - smoothed_handle[1]
                x0 = smoothed_handle[0]
                y0 = smoothed_handle[1]

                # 5) Draw line from handle → puck
                cv2.line(vis, (xh, yh), (xp, yp), (0, 255, 255), 3)

                # 6) Compute first‐wall bounce
                first_bounce = compute_first_bounce(x0, y0, vx, vy, TABLE_W, TABLE_H)
                if first_bounce is not None:
                    t1, xh1, yh1, w1 = first_bounce
                    # Draw first intersection
                    cv2.circle(vis, (int(round(xh1)), int(round(yh1))), 6, (255, 0, 0), -1)

                    # Reflect vector at first bounce
                    vx_ref, vy_ref = reflect_vector(vx, vy, w1)

                    # 7) Compute second bounce:
                    #    - Start just beyond the first intersection, so we don’t re‐hit the same wall at t≈0.
                    eps = 1e-3
                    x0b = xh1 + vx_ref * eps
                    y0b = yh1 + vy_ref * eps
                    second_bounce = compute_first_bounce(x0b, y0b, vx_ref, vy_ref, TABLE_W, TABLE_H)

                    if second_bounce is not None:
                        t2, xh2, yh2, w2 = second_bounce
                        # Draw second intersection
                        cv2.circle(vis, (int(round(xh2)), int(round(yh2))), 6, (0, 0, 255), -1)
                        # Draw line from first‐bounce → second‐bounce
                        cv2.line(vis,
                                 (int(round(xh1)), int(round(yh1))),
                                 (int(round(xh2)), int(round(yh2))),
                                 (255, 0, 255),
                                 3)
                    else:
                        # If no second bounce (e.g. nearly parallel?), just draw a short arrow from first bounce
                        end_pt = (
                            int(round(xh1 + vx_ref*100)),
                            int(round(yh1 + vy_ref*100))
                        )
                        cv2.line(vis,
                                 (int(round(xh1)), int(round(yh1))),
                                 end_pt,
                                 (255, 0, 255),
                                 3)

                    # Send data to STM32: "x_handle,y_handle,vx_ref,vy_ref\n"
                    if ser is not None:
                        msg = f"{x0:.2f},{y0:.2f},{vx_ref:.2f},{vy_ref:.2f}\n"
                        ser.write(msg.encode('ascii'))

                    # Annotate which wall was hit first
                    cv2.putText(vis, f"Hit: {w1.upper()}",
                                (30, TABLE_H - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Show the full visualization
        cv2.imshow(win, vis)
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
    parser = argparse.ArgumentParser(description="Air Hockey Table Detection on Raspberry Pi (smoothed, multi‐window HSV)")
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
