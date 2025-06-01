#!/usr/bin/env python3
# airhockey.py
#
# 1) MODE calibrate_frame: click two points per side (Top, Right, Bottom, Left) to compute a perspective warp matrix.
#    - Saves --> "warp_matrix.json" (with keys: "matrix", "width", "height")
# 2) MODE calibrate_hsv: click on all visible red circles to sample HSV values. Press 'q' when finished.
#    - Saves --> "hsv_ranges.json" (with keys: h_min,h_max,s_min,s_max,v_min,v_max)
# 3) MODE run: loads both JSON files, captures from Pi camera, warps, thresholds, HoughCircles, computes bounce, sends via serial.
#
# DEPENDENCIES: pip3 install opencv-python numpy pyserial
#
# USAGE:
#   python3 airhockey.py --mode calibrate_frame
#   python3 airhockey.py --mode calibrate_hsv
#   python3 airhockey.py --mode run
#
# SERIAL PROTOCOL: sends ASCII lines: "x0,y0,vx_ref,vy_ref\n"
#    where (x0,y0) is circle1 center in warped frame, and (vx_ref,vy_ref) is the reflected velocity after hitting first wall.
#
# To autostart on boot (see bottom), create a systemd service that runs:
#   /usr/bin/python3 /home/pi/airhockey.py --mode run
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

# How many clicks per side during frame calibration
CLICKS_PER_SIDE = 2

# Tolerances to expand from sampled HSV values
H_MARGIN = 10    # hue ±10 (0–180)
S_MARGIN = 40    # sat ±40 (0–255)
V_MARGIN = 40    # val ±40 (0–255)

# ------------------------------------------------------------------------------
# GLOBALS FOR MOUSE CALLBACKS
# ------------------------------------------------------------------------------
clicks     = []      # raw pixel coords during calibration
hsv_samples = []     # list of (h,s,v) tuples

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
    # a = y1 - y2
    # b = x2 - x1
    # c = x1*y2 - x2*y1
    a = float(y1 - y2)
    b = float(x2 - x1)
    c = float(x1 * y2 - x2 * y1)
    return (a, b, c)

def intersect_lines(l1, l2):
    """
    Given l1=(a1,b1,c1) and l2=(a2,b2,c2), solve for their intersection:
        a1*x + b1*y + c1 = 0
        a2*x + b2*y + c2 = 0
    Returns (x, y) as floats.
    """
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    denom = (a1 * b2 - a2 * b1)
    if abs(denom) < 1e-8:
        # Lines are nearly parallel; return an average or default
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
        print(f"Sampled HSV at ({x},{y}):  H={h}, S={s}, V={v}")

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

    cv2.namedWindow("Frame Calibration")
    cv2.setMouseCallback("Frame Calibration", mouse_callback_frame)

    side_names = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    pts_per_side = []
    side_idx = 0

    print("== FRAME CALIBRATION ==")
    print("You will click 2 points on each side of the table frame, in order:")
    print("  1) TOP edge: click 2 points, then press 'n'")
    print("  2) RIGHT edge: click 2 points, then press 'n'")
    print("  3) BOTTOM edge: click 2 points, then press 'n'")
    print("  4) LEFT edge: click 2 points, then press 'n'")
    print("Press 'q' at any time to abort.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Draw existing clicks
        vis = frame.copy()
        for (x, y) in clicks:
            cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)

        cv2.putText(vis, f"Click 2 points on the {side_names[side_idx]} edge", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Frame Calibration", vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('n'):
            # Move to next side, but only if exactly 2 clicks were added for this side
            if len(clicks) < (side_idx+1)*CLICKS_PER_SIDE:
                print(f"  >> You must click exactly {CLICKS_PER_SIDE} points on the {side_names[side_idx]} edge before pressing 'n'.")
                continue
            side_idx += 1
            if side_idx >= 4:
                # Got all 8 points
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
        print(f"ERROR: Expected 8 total clicks (2 per side). Got {len(clicks)}. Aborting.")
        return

    # Group clicks by side: 0-1 = top, 2-3=right, 4-5=bottom, 6-7=left
    top_pts    = clicks[0:2]
    right_pts  = clicks[2:4]
    bottom_pts = clicks[4:6]
    left_pts   = clicks[6:8]

    # Compute line eqns for each side
    l_top    = line_from_two_points(top_pts[0],    top_pts[1])
    l_right  = line_from_two_points(right_pts[0],  right_pts[1])
    l_bottom = line_from_two_points(bottom_pts[0], bottom_pts[1])
    l_left   = line_from_two_points(left_pts[0],   left_pts[1])

    # Intersect to get 4 corners:
    # top-left = intersection of top and left
    # top-right = intersection of top and right
    # bottom-right = intersection of bottom and right
    # bottom-left = intersection of bottom and left
    tl = intersect_lines(l_top,    l_left)
    tr = intersect_lines(l_top,    l_right)
    br = intersect_lines(l_bottom, l_right)
    bl = intersect_lines(l_bottom, l_left)

    # Compute width, height in px for the destination rectangle
    widthA  = math.hypot(br[0]-bl[0], br[1]-bl[1])
    widthB  = math.hypot(tr[0]-tl[0], tr[1]-tl[1])
    maxWidth  = max(int(widthA), int(widthB))

    heightA = math.hypot(tr[0]-br[0], tr[1]-br[1])
    heightB = math.hypot(tl[0]-bl[0], tl[1]-bl[1])
    maxHeight = max(int(heightA), int(heightB))

    # Source and destination coordinates for warp
    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([[0,0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Save to JSON
    save_warp_matrix(FRAME_CALIB_FILE, M, maxWidth, maxHeight)
    print(f"Frame calibration saved to '{FRAME_CALIB_FILE}'.")
    print(f"  Table warped size = {maxWidth} x {maxHeight} px.")

# ------------------------------------------------------------------------------
# CALIBRATION: HSV RANGE
# ------------------------------------------------------------------------------

def calibrate_hsv():
    """
    Captures one frame from camera, lets user click on all visible red circles
    (any number of clicks). When user presses 'q', compute min/max HSV over all samples,
    expand by margins, clamp to valid range, and save to JSON.
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
        print("ERROR: Could not grab a frame.")
        return

    # Use the raw (unwarped) frame for HSV sampling
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    display = frame.copy()
    cv2.namedWindow("HSV Calibration")
    cv2.setMouseCallback("HSV Calibration", mouse_callback_hsv, {"hsv_frame": frame_hsv})

    print("== HSV CALIBRATION ==")
    print("Click repeatedly on visible red circles (only) in the image.")
    print("When done, press 'q' to finish.")

    while True:
        vis = display.copy()
        # Draw circles at each sampled point
        for (h, s, v) in hsv_samples:
            # We don't know x,y here, so skip drawing; just print counts
            pass
        cv2.putText(vis, f"Samples = {len(hsv_samples)}; press 'q' when done.", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("HSV Calibration", vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(hsv_samples) == 0:
        print("No HSV samples were collected. Aborting.")
        return

    # Compute min/max for each channel
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
    print(f"HSV calibration saved to '{HSV_CALIB_FILE}':")
    print(f"  H ∈ [{h_min}, {h_max}],  S ∈ [{s_min}, {s_max}],  V ∈ [{v_min}, {v_max}]")

# ------------------------------------------------------------------------------
# MAIN DETECTION + SERIAL SENDING
# ------------------------------------------------------------------------------

def reflect_vector(vx, vy, hit_wall):
    """
    Reflect (vx, vy) across the wall. hit_wall is one of 'left','right','top','bottom'.
    Returns (vx_ref, vy_ref).
    """
    if hit_wall in ("left","right"):
        return (-vx, vy)
    elif hit_wall in ("top","bottom"):
        return (vx, -vy)
    else:
        return (vx, vy)

def compute_first_bounce(x0, y0, vx, vy, W, H):
    """
    Given a ray starting at (x0,y0) with direction (vx,vy), compute which wall it hits first.
    Walls: x=0 ('left'), x=W ('right'), y=0 ('top'), y=H ('bottom').
    Returns (t_hit, x_hit, y_hit, wall_name). If no positive intersection, returns None.
    """
    candidates = []
    # LEFT wall (x=0) if vx<0
    if vx < 0:
        t = (0 - x0) / vx
        if t > 0:
            y_hit = y0 + t*vy
            if 0 <= y_hit <= H:
                candidates.append((t, 0.0, y_hit, "left"))
    # RIGHT wall (x=W) if vx>0
    if vx > 0:
        t = (W - x0) / vx
        if t > 0:
            y_hit = y0 + t*vy
            if 0 <= y_hit <= H:
                candidates.append((t, float(W), y_hit, "right"))
    # TOP wall (y=0) if vy<0
    if vy < 0:
        t = (0 - y0) / vy
        if t > 0:
            x_hit = x0 + t*vx
            if 0 <= x_hit <= W:
                candidates.append((t, x_hit, 0.0, "top"))
    # BOTTOM wall (y=H) if vy>0
    if vy > 0:
        t = (H - y0) / vy
        if t > 0:
            x_hit = x0 + t*vx
            if 0 <= x_hit <= W:
                candidates.append((t, x_hit, float(H), "bottom"))

    if not candidates:
        return None

    # pick the smallest positive t
    t_hit, xh, yh, wall = min(candidates, key=lambda e: e[0])
    return (t_hit, xh, yh, wall)

def main_loop():
    """
    Loads warp matrix + HSV thresholds, opens serial port, and runs main detection loop:
    - Grab frame, warp, threshold for red, HoughCircles, pick two largest, compute vector, reflect, send via serial.
    - Displays diagnostic window with circles, bounce point, reflected ray.
    """
    # Check that calibration files exist
    if not os.path.exists(FRAME_CALIB_FILE):
        print(f"ERROR: Frame calibration file '{FRAME_CALIB_FILE}' not found. Run --mode calibrate_frame first.")
        return
    if not os.path.exists(HSV_CALIB_FILE):
        print(f"ERROR: HSV calibration file '{HSV_CALIB_FILE}' not found. Run --mode calibrate_hsv first.")
        return

    # Load calibration
    warp_matrix, TABLE_W, TABLE_H = load_warp_matrix(FRAME_CALIB_FILE)
    hsv_lower, hsv_upper       = load_hsv_ranges(HSV_CALIB_FILE)

    # Open serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2.0)  # allow STM32 to reset if necessary
        print(f"Opened serial on {SERIAL_PORT} @ {BAUD_RATE} baud.")
    except Exception as e:
        print(f"WARNING: Could not open serial port '{SERIAL_PORT}': {e}")
        ser = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera for main loop.")
        return

    cv2.namedWindow("AirHockey Detection")

    print("== RUNNING DETECTION (press 'q' to quit) ==")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1) Warp to table view
        warped = cv2.warpPerspective(frame, warp_matrix, (TABLE_W, TABLE_H))

        # 2) Convert to HSV, threshold
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        # 3) Blur + find circles
        gray = cv2.bitwise_and(warped, warped, mask=mask)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
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

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Sort by radius descending, keep top-2
            circles = sorted(circles, key=lambda c: c[2], reverse=True)[:2]
            if len(circles) >= 2:
                (x1, y1, r1) = circles[0]
                (x2, y2, r2) = circles[1]

                # Draw circles
                cv2.circle(vis, (x1, y1), r1, (0, 255, 0), 2)
                cv2.circle(vis, (x1, y1), 2, (0, 0, 255), 3)
                cv2.circle(vis, (x2, y2), r2, (0, 255, 0), 2)
                cv2.circle(vis, (x2, y2), 2, (0, 0, 255), 3)

                # Compute vector from circle1 to circle2
                vx = float(x2 - x1)
                vy = float(y2 - y1)
                x0 = float(x1)
                y0 = float(y1)

                # Compute first bounce against table bounds
                bounce = compute_first_bounce(x0, y0, vx, vy, TABLE_W, TABLE_H)

                if bounce is not None:
                    t_hit, xh, yh, wall = bounce
                    # Draw intersection point
                    cv2.circle(vis, (int(round(xh)), int(round(yh))), 5, (255, 0, 0), -1)
                    # Reflect vector
                    vx_ref, vy_ref = reflect_vector(vx, vy, wall)
                    # Draw reflected ray (for visualization use scaled-up by 100 px)
                    pt_hit = np.array([xh, yh], dtype=np.float32)
                    dir_ref = np.array([vx_ref, vy_ref], dtype=np.float32)
                    norm = np.linalg.norm(dir_ref)
                    if norm > 1e-3:
                        dir_ref_unit = dir_ref / norm
                        end_pt = (pt_hit + dir_ref_unit * 100.0).astype(int)
                        cv2.line(vis,
                                 (int(round(xh)), int(round(yh))),
                                 (int(end_pt[0]), int(end_pt[1])),
                                 (255, 0, 255), 2)

                    # Send via serial: "x0,y0,vx_ref,vy_ref\n"
                    if ser is not None:
                        msg = f"{x0:.2f},{y0:.2f},{vx_ref:.2f},{vy_ref:.2f}\n"
                        ser.write(msg.encode('ascii'))

                    # Overlay text
                    cv2.putText(vis, f"Wall={wall}", (10, TABLE_H - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("AirHockey Detection", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()
        print("Serial port closed.")

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
