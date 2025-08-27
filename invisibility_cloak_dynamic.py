# invisibility_cloak_dynamic.py
import cv2
import numpy as np
import time

# Global variable to let mouse callback access the latest HSV frame
last_hsv = None

def nothing(x):
    pass

def capture_background(cap, num_frames=60, resize=(640,480)):
    """Capture several frames and return median background (reduces transient noise)."""
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    if len(frames) == 0:
        return None
    bg = np.median(frames, axis=0).astype(dtype=np.uint8)
    return bg

def sample_color_callback(event, x, y, flags, param):
    """When user left-clicks a pixel in the preview, sample HSV around that point and set trackbars."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    global last_hsv
    if last_hsv is None:
        print("[sample] No frame available yet.")
        return
    h, w = last_hsv.shape[:2]
    # small ROI around cursor
    x1, y1 = max(0, x-5), max(0, y-5)
    x2, y2 = min(w, x+5), min(h, y+5)
    roi = last_hsv[y1:y2, x1:x2]
    if roi.size == 0:
        return
    mean_hsv = np.mean(roi.reshape(-1,3), axis=0).astype(int)
    h_mean, s_mean, v_mean = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

    # default tolerances (you can tune these or use trackbars)
    H_tol = 10
    S_tol = 50
    V_tol = 50

    H_low = max(0, h_mean - H_tol)
    H_high = min(179, h_mean + H_tol)
    S_low = max(0, s_mean - S_tol)
    S_high = min(255, s_mean + S_tol)
    V_low = max(0, v_mean - V_tol)
    V_high = min(255, v_mean + V_tol)

    cv2.setTrackbarPos('H_low', 'Controls', H_low)
    cv2.setTrackbarPos('H_high','Controls', H_high)
    cv2.setTrackbarPos('S_low', 'Controls', S_low)
    cv2.setTrackbarPos('S_high','Controls', S_high)
    cv2.setTrackbarPos('V_low', 'Controls', V_low)
    cv2.setTrackbarPos('V_high','Controls', V_high)

    print(f"[sample] HSV mean: ({h_mean},{s_mean},{v_mean}) -> ranges H:{H_low}-{H_high} S:{S_low}-{S_high} V:{V_low}-{V_high}")

def get_trackbar_values():
    hl = cv2.getTrackbarPos('H_low','Controls')
    hh = cv2.getTrackbarPos('H_high','Controls')
    sl = cv2.getTrackbarPos('S_low','Controls')
    sh = cv2.getTrackbarPos('S_high','Controls')
    vl = cv2.getTrackbarPos('V_low','Controls')
    vh = cv2.getTrackbarPos('V_high','Controls')
    return (hl, hh, sl, sh, vl, vh)

def main():
    global last_hsv
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Preferred working resolution (smaller => faster)
    W, H = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    # Create UI windows
    cv2.namedWindow('Invisibility', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 400, 280)

    # Trackbars: ranges are H:0-179, S:0-255, V:0-255
    cv2.createTrackbar('H_low', 'Controls', 0, 179, nothing)
    cv2.createTrackbar('H_high','Controls', 179, 179, nothing)
    cv2.createTrackbar('S_low', 'Controls', 50, 255, nothing)
    cv2.createTrackbar('S_high','Controls', 255, 255, nothing)
    cv2.createTrackbar('V_low', 'Controls', 50, 255, nothing)
    cv2.createTrackbar('V_high','Controls', 255, 255, nothing)

    # mouse callback to sample color
    cv2.setMouseCallback('Invisibility', sample_color_callback)

    # initial background capture
    print("Capturing background... Please make sure nobody is in front of the camera.")
    background = capture_background(cap, num_frames=60, resize=(W,H))
    if background is None:
        print("Warning: Failed to capture background. Background will be black until you press 'r'.")
        background = np.zeros((H,W,3), dtype=np.uint8)
    else:
        print("Background captured. Now show cloth to camera and click to sample color, or press 'a' to auto-calibrate.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (W,H))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        last_hsv = hsv.copy()

        hl, hh, sl, sh, vl, vh = get_trackbar_values()

        # handle hue wrap-around (red region)
        if hl <= hh:
            lower = np.array([hl, sl, vl])
            upper = np.array([hh, sh, vh])
            mask = cv2.inRange(hsv, lower, upper)
        else:
            # example: hl=170, hh=10 -> two ranges
            lower1 = np.array([0, sl, vl])
            upper1 = np.array([hh, sh, vh])
            lower2 = np.array([hl, sl, vl])
            upper2 = np.array([179, sh, vh])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleaning
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (7,7), 0)

        inv_mask = cv2.bitwise_not(mask)

        # Use the captured background
        bg = cv2.resize(background, (W,H))

        # Compose
        cloak_area = cv2.bitwise_and(bg, bg, mask=mask)
        non_cloak_area = cv2.bitwise_and(frame, frame, mask=inv_mask)
        final = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

        # Display
        cv2.imshow('Invisibility', final)
        cv2.imshow('Mask', mask)
        cv2.imshow('Original', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # re-capture background
            print("[r] Re-capturing background. Make sure scene is clear.")
            background = capture_background(cap, num_frames=60, resize=(W,H))
            if background is not None:
                print("[r] Background updated.")
        elif key == ord('a'):
            # auto-calibrate: capture several frames while user holds the cloth near center
            print("[a] Auto-calibrating: hold cloth near center for ~1.5s...")
            samples = []
            for i in range(30):
                ret2, f2 = cap.read()
                if not ret2:
                    continue
                f2 = cv2.flip(f2,1)
                f2 = cv2.resize(f2, (W,H))
                h2 = cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)
                cx, cy = W//2, H//2
                roi = h2[cy-30:cy+30, cx-30:cx+30]
                if roi.size == 0:
                    continue
                samples.append(roi.reshape(-1,3))
                # small sleep to allow frames to update
                time.sleep(0.03)
            if len(samples) > 0:
                samples = np.concatenate(samples, axis=0)
                mean = np.mean(samples, axis=0).astype(int)
                h_mean, s_mean, v_mean = int(mean[0]), int(mean[1]), int(mean[2])
                H_tol = 10; S_tol = 50; V_tol = 50
                H_low = max(0, h_mean - H_tol)
                H_high = min(179, h_mean + H_tol)
                S_low = max(0, s_mean - S_tol)
                S_high = min(255, s_mean + S_tol)
                V_low = max(0, v_mean - V_tol)
                V_high = min(255, v_mean + V_tol)
                cv2.setTrackbarPos('H_low','Controls', H_low)
                cv2.setTrackbarPos('H_high','Controls', H_high)
                cv2.setTrackbarPos('S_low','Controls', S_low)
                cv2.setTrackbarPos('S_high','Controls', S_high)
                cv2.setTrackbarPos('V_low','Controls', V_low)
                cv2.setTrackbarPos('V_high','Controls', V_high)
                print(f"[a] Auto-calibrated to H:{H_low}-{H_high} S:{S_low}-{S_high} V:{V_low}-{V_high}")
            else:
                print("[a] Auto-calibration failed — no samples collected.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
