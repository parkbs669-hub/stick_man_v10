"""
Tennis Stickman Animation Generator v2
Downloads tennis video -> MediaPipe Tasks Pose 33 landmarks -> Custom Stickman with Pentagon Torso, Red Racket & Gray Shoes -> Save as MP4.

Modifications from v1 (based on reference stickman image):
  1. Pentagon torso (filled polygon) instead of line-based torso
  2. Larger head (~80% of shoulder width)
  3. Thicker body lines (7*SSAA), no visible joint circles
  4. Dark gray hands (grip style)
  5. Improved racket: larger red ellipse, minimal strings, short grip
  6. Slightly larger shoes with more presence
  7. Overall line weight & outline thickness increased
"""

import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
import math
import urllib.request
from types import SimpleNamespace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────

VIDEO_URL = "https://youtu.be/J4WFqsoGDgg?si=RtXWDJE_i5jtEUmv"
INPUT_VIDEO = "federer_serve_input.mp4"
OUTPUT_VIDEO = "federer_serve_stickman.mp4"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker_full.task"

# Super-Sampling Anti-Aliasing: render at SSAA× resolution, downscale for smooth lines
SSAA = 2

# ── Stickman Style Parameters (v2 tweaks) ──
BODY_COLOR = (20, 20, 20)              # Deep black
BODY_THICKNESS = 7 * SSAA              # ↑ thicker than v1 (was 5*SSAA)
HEAD_FILL_COLOR = (255, 255, 255)      # Pure white
OUTLINE_COLOR = (20, 20, 20)           # Outline black
OUTLINE_THICKNESS = 6 * SSAA           # ↑ thicker outlines (was 5*SSAA)
SHOE_FILL_COLOR = (155, 155, 155)      # Gray shoe
SHOE_OUTLINE_COLOR = (20, 20, 20)      # Pure black outline
HAND_FILL_COLOR = (60, 60, 60)         # ★ NEW: dark gray hands (was white)
HAND_OUTLINE_COLOR = (20, 20, 20)      # Black outline for hands

# ── Racket Style Parameters (v2 tweaks) ──
RACKET_FRAME_COLOR = (30, 30, 220)     # Bright red (BGR)
RACKET_STRING_COLOR = (180, 180, 220)  # Very faint pinkish gray
RACKET_GRIP_COLOR = (40, 40, 40)       # Dark gray/black

# ── Court Style Parameters ──
COURT_GREEN = (78, 115, 76)            # Classic grass/hardcourt green (BGR)
SKY_GRADIENT_START = (215, 215, 215)   # Light gray at top (BGR)
SKY_GRADIENT_END = (238, 238, 238)     # Lighter gray at horizon (BGR)

# ── Landmark smoothing (EMA): 0 = max smooth/laggy, 1 = no smooth/jittery ──
EMA_ALPHA = 0.55

# ── Stickman Alignment Tuning ──
SCALE_FACTOR = 0.9
OFFSET_X = 0
OFFSET_Y = 10

# ── Landmark Indices (MediaPipe Pose 33) ──
NOSE = 0
L_SHOULDER = 11; R_SHOULDER = 12
L_ELBOW = 13;    R_ELBOW = 14
L_WRIST = 15;    R_WRIST = 16
L_HIP = 23;      R_HIP = 24
L_KNEE = 25;     R_KNEE = 26
L_ANKLE = 27;    R_ANKLE = 28
L_HEEL = 29;     R_HEEL = 30
L_FOOT_INDEX = 31; R_FOOT_INDEX = 32

# ─────────────────────────────────────────
# Step 1: Download Video & Model
# ─────────────────────────────────────────

def download_video():
    """Downloads the tennis video using yt-dlp."""
    if os.path.exists(INPUT_VIDEO):
        print(f"[✓] Input video already exists: {INPUT_VIDEO}")
        return True
    print(f"[↓] Downloading video from {VIDEO_URL} ...")
    try:
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", INPUT_VIDEO,
            VIDEO_URL,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[✓] Video downloaded: {INPUT_VIDEO}")
        return True
    except Exception as e:
        print(f"[✗] Failed to download video: {e}")
        return False


def download_model():
    """Downloads the MediaPipe Pose Landmarker model."""
    if os.path.exists(MODEL_PATH):
        print(f"[✓] MediaPipe model already exists: {MODEL_PATH}")
        return True
    print(f"[↓] Downloading MediaPipe model ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[✓] Model downloaded: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[✗] Failed to download model: {e}")
        return False


# ─────────────────────────────────────────
# Step 2: Perspective Projection & Background
# ─────────────────────────────────────────

def get_projected_pt(cx, cy, w, h):
    """
    Projects a normalized court coordinate (cx, cy) to screen pixels.
    cx in [-1.0, 1.0] (doubles boundaries)
    cy in [0.0, 1.0]  (0.0: net/horizon, 1.0: baseline/bottom)
    """
    horizon_y = int(h * 0.6)
    screen_y = horizon_y + int(cy * (h - horizon_y))
    perspective_scale = 0.3 + 0.7 * cy
    screen_x = int(w / 2 + cx * (w / 2) * perspective_scale)
    return screen_x, screen_y


def draw_court_background(w, h):
    """Draws a beautiful 3D perspective tennis court with gradient sky."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    horizon_y = int(h * 0.6)

    # Sky gradient (top to horizon)
    for y in range(horizon_y):
        t = y / max(horizon_y - 1, 1)
        color = tuple(
            int(SKY_GRADIENT_START[c] * (1 - t) + SKY_GRADIENT_END[c] * t)
            for c in range(3)
        )
        bg[y, :] = color

    # Court surface gradient (horizon to bottom)
    court_top = np.array(COURT_GREEN, dtype=np.float32)
    court_bot = np.array(COURT_GREEN, dtype=np.float32) * 0.75
    for y in range(horizon_y, h):
        t = (y - horizon_y) / max(h - horizon_y - 1, 1)
        color = tuple(int(court_top[c] * (1 - t) + court_bot[c] * t) for c in range(3))
        bg[y, :] = color

    # Court lines
    line_color = (200, 200, 200)
    line_thick = max(1, 2 * SSAA // 2)

    # Baseline
    bl_l = get_projected_pt(-0.9, 0.95, w, h)
    bl_r = get_projected_pt(0.9, 0.95, w, h)
    cv2.line(bg, bl_l, bl_r, line_color, line_thick, cv2.LINE_AA)

    # Service line
    sl_l = get_projected_pt(-0.9, 0.55, w, h)
    sl_r = get_projected_pt(0.9, 0.55, w, h)
    cv2.line(bg, sl_l, sl_r, line_color, line_thick, cv2.LINE_AA)

    # Center service line
    cs_t = get_projected_pt(0.0, 0.0, w, h)
    cs_b = get_projected_pt(0.0, 0.55, w, h)
    cv2.line(bg, cs_t, cs_b, line_color, line_thick, cv2.LINE_AA)

    # Side lines (singles)
    for x_norm in [-0.7, 0.7]:
        top_pt = get_projected_pt(x_norm, 0.0, w, h)
        bot_pt = get_projected_pt(x_norm, 0.95, w, h)
        cv2.line(bg, top_pt, bot_pt, line_color, line_thick, cv2.LINE_AA)

    # Side lines (doubles)
    for x_norm in [-0.9, 0.9]:
        top_pt = get_projected_pt(x_norm, 0.0, w, h)
        bot_pt = get_projected_pt(x_norm, 0.95, w, h)
        cv2.line(bg, top_pt, bot_pt, line_color, line_thick, cv2.LINE_AA)

    return bg


# ─────────────────────────────────────────
# Step 3: Stickman Components Drawing
# ─────────────────────────────────────────

def get_point(landmarks, idx, w, h):
    """Retrieves 2D coordinates scaled and shifted according to config."""
    lm = landmarks[idx]
    x = lm.x * w
    y = lm.y * h
    # Apply scale around center
    cx, cy = w / 2, h / 2
    x = cx + (x - cx) * SCALE_FACTOR + OFFSET_X * SSAA
    y = cy + (y - cy) * SCALE_FACTOR + OFFSET_Y * SSAA
    return (int(x), int(y))


def draw_head(canvas, cx, cy, radius):
    """Draws a clean white circular head with a very thick outline."""
    cv2.circle(canvas, (cx, cy), radius, HEAD_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius, OUTLINE_COLOR, OUTLINE_THICKNESS, cv2.LINE_AA)


def draw_body_line(canvas, p1, p2, thickness=None):
    """Draws a thick body line for stickman limbs with round caps."""
    if thickness is None:
        thickness = BODY_THICKNESS
    cv2.line(canvas, p1, p2, BODY_COLOR, thickness, cv2.LINE_AA)


def draw_pentagon_torso(canvas, neck, l_shoulder, r_shoulder, l_hip, r_hip):
    """
    ★ NEW in v2: Draws the torso as a filled black pentagon.
    Vertices: neck (top) -> right_shoulder -> right_hip -> left_hip -> left_shoulder
    This matches the reference image where the torso is a solid polygon shape.
    """
    pts = np.array([neck, r_shoulder, r_hip, l_hip, l_shoulder], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], BODY_COLOR, cv2.LINE_AA)
    cv2.polylines(canvas, [pts], isClosed=True, color=OUTLINE_COLOR,
                  thickness=OUTLINE_THICKNESS, lineType=cv2.LINE_AA)


def draw_hand(canvas, wrist, radius):
    """
    ★ v2: Draws a DARK gray circle for the hand (like a gripping glove).
    Changed from white to dark gray to match reference image.
    """
    cv2.circle(canvas, wrist, radius, HAND_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, wrist, radius, HAND_OUTLINE_COLOR, max(2 * SSAA, 2), cv2.LINE_AA)


def draw_shoe(canvas, ankle, heel, toe):
    """
    Draws a realistic, organic tennis shoe at the foot position.
    Uses ankle, heel, and toe landmarks for proper orientation and shape.
    v2: slightly larger for more visual presence.
    """
    sole_vec = np.array(toe) - np.array(heel)
    L = np.linalg.norm(sole_vec) + 1e-6
    u_sole = sole_vec / L

    perp = np.array([-u_sole[1], u_sole[0]])  # perpendicular (upward)

    shoe_length = L * 1.4   # ↑ slightly longer (was ~1.2 implied)
    shoe_height = L * 0.55  # ↑ slightly taller

    base_center = (np.array(heel, dtype=np.float64) + np.array(toe, dtype=np.float64)) / 2.0

    # Shoe outline polygon (6 points)
    p_heel_bot = base_center - u_sole * shoe_length * 0.45
    p_toe_bot = base_center + u_sole * shoe_length * 0.55
    p_toe_top = p_toe_bot + perp * shoe_height * 0.4
    p_mid_top = base_center + perp * shoe_height
    p_ankle_top = np.array(ankle, dtype=np.float64) + perp * shoe_height * 0.2
    p_heel_top = p_heel_bot + perp * shoe_height * 0.7

    pts = np.array([p_heel_bot, p_toe_bot, p_toe_top, p_mid_top, p_ankle_top, p_heel_top],
                   dtype=np.int32)

    cv2.fillPoly(canvas, [pts], SHOE_FILL_COLOR, cv2.LINE_AA)
    cv2.polylines(canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                  thickness=max(3 * SSAA, 2), lineType=cv2.LINE_AA)


def draw_shadow(canvas, l_ankle, r_ankle):
    """Draws a soft elliptical drop shadow beneath the feet. v2: reduced opacity."""
    cx = (l_ankle[0] + r_ankle[0]) // 2
    cy = max(l_ankle[1], r_ankle[1]) + 6 * SSAA
    spread = max(abs(l_ankle[0] - r_ankle[0]), 40 * SSAA)
    rx = int(spread * 0.7)
    ry = max(int(spread * 0.1), 6 * SSAA)

    overlay = canvas.copy()
    cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
    alpha = 0.15  # ↓ reduced from ~0.25 for subtler shadow
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def draw_racket(canvas, wrist, elbow, head_r):
    """
    ★ v2 improved racket:
    - Larger red ellipse frame (~1.3x head_r)
    - Minimal/no visible strings
    - Short thick black grip
    - Size scaled relative to head_r for 3D stability
    """
    wx, wy = wrist
    ex, ey = elbow
    dx, dy = wx - ex, wy - ey
    arm_len = math.sqrt(dx * dx + dy * dy) + 1e-6
    nx, ny = dx / arm_len, dy / arm_len  # direction from elbow to wrist (outward)

    # Racket dimensions based on head radius
    grip_length = int(head_r * 0.9)
    frame_rx = int(head_r * 1.0)     # horizontal radius of racket head ellipse
    frame_ry = int(head_r * 1.35)    # vertical radius (taller ellipse)
    frame_thickness = max(int(4 * SSAA), 3)

    # Grip start = wrist, grip end extends outward
    grip_end_x = int(wx + nx * grip_length)
    grip_end_y = int(wy + ny * grip_length)

    # Draw grip (thick black line)
    cv2.line(canvas, (wx, wy), (grip_end_x, grip_end_y),
             RACKET_GRIP_COLOR, max(int(5 * SSAA), 4), cv2.LINE_AA)

    # Racket head center = beyond grip end
    head_cx = int(grip_end_x + nx * frame_ry)
    head_cy = int(grip_end_y + ny * frame_ry)

    # Angle of the racket
    angle = math.degrees(math.atan2(ny, nx))

    # Draw racket head (red ellipse frame)
    cv2.ellipse(canvas, (head_cx, head_cy), (frame_rx, frame_ry),
                angle, 0, 360, RACKET_FRAME_COLOR, frame_thickness, cv2.LINE_AA)

    # Minimal strings: just 2 cross lines (very faint)
    string_thick = max(1, SSAA)
    perp_x, perp_y = -ny, nx  # perpendicular to racket direction
    for frac in [-0.3, 0.0, 0.3]:
        sx = int(head_cx + perp_x * frame_rx * frac * 0.8)
        sy = int(head_cy + perp_y * frame_rx * frac * 0.8)
        s1x = int(sx - nx * frame_ry * 0.6)
        s1y = int(sy - ny * frame_ry * 0.6)
        s2x = int(sx + nx * frame_ry * 0.6)
        s2y = int(sy + ny * frame_ry * 0.6)
        cv2.line(canvas, (s1x, s1y), (s2x, s2y), RACKET_STRING_COLOR, string_thick, cv2.LINE_AA)

    for frac in [-0.3, 0.0, 0.3]:
        sx = int(head_cx + nx * frame_ry * frac * 0.8)
        sy = int(head_cy + ny * frame_ry * frac * 0.8)
        s1x = int(sx - perp_x * frame_rx * 0.6)
        s1y = int(sy - perp_y * frame_rx * 0.6)
        s2x = int(sx + perp_x * frame_rx * 0.6)
        s2y = int(sy + perp_y * frame_rx * 0.6)
        cv2.line(canvas, (s1x, s1y), (s2x, s2y), RACKET_STRING_COLOR, string_thick, cv2.LINE_AA)


def draw_stickman(canvas, landmarks, w, h, is_right_handed=True):
    """
    Assembles all components to render the full stickman pose.
    v2 draw order: shadow -> legs -> shoes -> pentagon torso -> arms -> hands -> racket -> head
    """
    lm = landmarks

    # Get all key points
    nose = get_point(lm, NOSE, w, h)
    l_shoulder = get_point(lm, L_SHOULDER, w, h)
    r_shoulder = get_point(lm, R_SHOULDER, w, h)
    l_elbow = get_point(lm, L_ELBOW, w, h)
    r_elbow = get_point(lm, R_ELBOW, w, h)
    l_wrist = get_point(lm, L_WRIST, w, h)
    r_wrist = get_point(lm, R_WRIST, w, h)
    l_hip = get_point(lm, L_HIP, w, h)
    r_hip = get_point(lm, R_HIP, w, h)
    l_knee = get_point(lm, L_KNEE, w, h)
    r_knee = get_point(lm, R_KNEE, w, h)
    l_ankle = get_point(lm, L_ANKLE, w, h)
    r_ankle = get_point(lm, R_ANKLE, w, h)
    l_heel = get_point(lm, L_HEEL, w, h)
    r_heel = get_point(lm, R_HEEL, w, h)
    l_foot_idx = get_point(lm, L_FOOT_INDEX, w, h)
    r_foot_idx = get_point(lm, R_FOOT_INDEX, w, h)

    # Derived points
    neck = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
    mid_hip = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)

    # Head radius: ~80% of shoulder width (v2: enlarged)
    shoulder_width = math.sqrt(
        (l_shoulder[0] - r_shoulder[0]) ** 2 + (l_shoulder[1] - r_shoulder[1]) ** 2
    )
    head_r = max(int(shoulder_width * 0.42), 12 * SSAA)

    # Head center: above neck
    head_cx = neck[0]
    head_cy = neck[1] - int(head_r * 1.05)

    # Hand radius (v2: slightly smaller)
    hand_r = max(int(head_r * 0.22), 4 * SSAA)

    # ── Draw order ──

    # 1) Shadow
    draw_shadow(canvas, l_ankle, r_ankle)

    # 2) Legs: hip -> knee -> ankle
    draw_body_line(canvas, l_hip, l_knee)
    draw_body_line(canvas, l_knee, l_ankle)
    draw_body_line(canvas, r_hip, r_knee)
    draw_body_line(canvas, r_knee, r_ankle)

    # 3) Shoes
    draw_shoe(canvas, l_ankle, l_heel, l_foot_idx)
    draw_shoe(canvas, r_ankle, r_heel, r_foot_idx)

    # 4) Pentagon torso ★
    draw_pentagon_torso(canvas, neck, l_shoulder, r_shoulder, l_hip, r_hip)

    # 5) Spine (neck to mid_hip) — hidden by pentagon, but ensures connectivity
    # Not drawn separately since pentagon covers it

    # 6) Arms: shoulder -> elbow -> wrist
    draw_body_line(canvas, l_shoulder, l_elbow)
    draw_body_line(canvas, l_elbow, l_wrist)
    draw_body_line(canvas, r_shoulder, r_elbow)
    draw_body_line(canvas, r_elbow, r_wrist)

    # 7) Hands (dark gray)
    draw_hand(canvas, l_wrist, hand_r)
    draw_hand(canvas, r_wrist, hand_r)

    # 8) Racket (in the dominant hand)
    if is_right_handed:
        draw_racket(canvas, r_wrist, r_elbow, head_r)
    else:
        draw_racket(canvas, l_wrist, l_elbow, head_r)

    # 9) Neck line (from pentagon top to head bottom)
    neck_top = (head_cx, head_cy + head_r)
    draw_body_line(canvas, neck, neck_top, thickness=BODY_THICKNESS)

    # 10) Head (drawn last so it's on top)
    draw_head(canvas, head_cx, head_cy, head_r)


# ─────────────────────────────────────────
# Step 4: Video Processing Loop
# ─────────────────────────────────────────

def process_video():
    """Main processing: read video -> detect pose -> draw stickman -> write output."""
    download_model()

    if not os.path.exists(INPUT_VIDEO):
        if not download_video():
            print("[✗] Cannot proceed without input video.")
            return

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[✗] Cannot open video: {INPUT_VIDEO}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    render_w = orig_w * SSAA
    render_h = orig_h * SSAA

    print(f"[i] Video: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames")
    print(f"[i] Rendering at {render_w}x{render_h} (SSAA={SSAA}x)")

    # Background (rendered once at SSAA resolution)
    court_bg = draw_court_background(render_w, render_h)

    # MediaPipe Pose Landmarker setup
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Output video writer (writes at original resolution)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (orig_w, orig_h))

    # EMA smoothing state
    smoothed = None  # will hold list of SimpleNamespace(x, y, z) for 33 landmarks

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000.0 / fps)

        # Convert to MediaPipe image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect pose
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        canvas = court_bg.copy()

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            raw = result.pose_landmarks[0]

            # EMA smoothing
            if smoothed is None:
                smoothed = [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z) for lm in raw]
            else:
                for i in range(len(raw)):
                    smoothed[i].x = EMA_ALPHA * raw[i].x + (1 - EMA_ALPHA) * smoothed[i].x
                    smoothed[i].y = EMA_ALPHA * raw[i].y + (1 - EMA_ALPHA) * smoothed[i].y
                    smoothed[i].z = EMA_ALPHA * raw[i].z + (1 - EMA_ALPHA) * smoothed[i].z

            draw_stickman(canvas, smoothed, render_w, render_h, is_right_handed=True)

        # Downscale from SSAA resolution to original
        final = cv2.resize(canvas, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        out.write(final)

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"  Processing: frame {frame_idx}/{total_frames} ({pct:.1f}%)")

    cap.release()
    out.release()
    landmarker.close()

    print(f"\n[✓] Stickman video saved: {OUTPUT_VIDEO}")
    print(f"    Resolution: {orig_w}x{orig_h} @ {fps:.1f}fps, {frame_idx} frames")


# ─────────────────────────────────────────
# Main Execution Entry Point
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TENNIS STICKMAN ANIMATION GENERATOR v2")
    print("  (Pentagon Torso + Enhanced Style)")
    print("=" * 60)
    process_video()
