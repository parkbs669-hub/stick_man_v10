"""
Tennis Stickman Animation Generator v3
사용법:
  python tennis_stickman_v3.py <YouTube_URL_또는_로컬파일> <동작명> [--left]

예시:
  python tennis_stickman_v3.py https://youtu.be/xxxx federer_serve
  python tennis_stickman_v3.py C:/Users/.../federer_input.mp4 federer_serve
  python tennis_stickman_v3.py https://youtu.be/xxxx nadal_forehand --left
"""

import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
import math
import urllib.request
import argparse
from types import SimpleNamespace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────
# CLI 인자 파싱
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="테니스 스틱맨 애니메이션 생성기 v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python tennis_stickman_v3.py https://youtu.be/xxxx federer_serve
  python tennis_stickman_v3.py https://youtu.be/xxxx nadal_forehand --left
  python tennis_stickman_v3.py https://youtu.be/xxxx djokovic_backhand
        """
    )
    parser.add_argument("url",  help="YouTube 영상 URL")
    parser.add_argument("name", help="동작명 (파일명에 사용, 예: federer_serve, nadal_forehand)")
    parser.add_argument("--left", action="store_true", help="왼손잡이 선수 (기본: 오른손잡이)")
    return parser.parse_args()


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker_full.task"

SSAA = 2

BODY_COLOR        = (20, 20, 20)
BODY_THICKNESS    = 7 * SSAA
HEAD_FILL_COLOR   = (255, 255, 255)
OUTLINE_COLOR     = (20, 20, 20)
OUTLINE_THICKNESS = 6 * SSAA
SHOE_FILL_COLOR   = (155, 155, 155)
SHOE_OUTLINE_COLOR = (20, 20, 20)
HAND_FILL_COLOR   = (60, 60, 60)
HAND_OUTLINE_COLOR = (20, 20, 20)

RACKET_FRAME_COLOR  = (30, 30, 220)
RACKET_STRING_COLOR = (180, 180, 220)
RACKET_GRIP_COLOR   = (40, 40, 40)

COURT_GREEN         = (78, 115, 76)
SKY_GRADIENT_START  = (215, 215, 215)
SKY_GRADIENT_END    = (238, 238, 238)

EMA_ALPHA          = 0.55
VISIBILITY_THRESHOLD = 0.5   # 이 값 미만인 랜드마크는 화면 밖으로 판단, 해당 부위 스킵
SCALE_FACTOR = 0.9
OFFSET_X     = 0
OFFSET_Y     = 10

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

def download_video(video_url, input_path):
    # 로컬 파일 경로인 경우 그대로 사용
    if not video_url.startswith("http"):
        if os.path.exists(video_url):
            print(f"[✓] Using local file: {video_url}")
            return video_url
        print(f"[✗] Local file not found: {video_url}")
        return None

    if os.path.exists(input_path):
        print(f"[✓] Input video already exists: {input_path}")
        return input_path
    print(f"[↓] Downloading video from {video_url} ...")
    try:
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", input_path,
            video_url,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[✓] Video downloaded: {input_path}")
        return input_path
    except Exception as e:
        print(f"[✗] Failed to download video: {e}")
        return None


def download_model():
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
# Step 2: Background
# ─────────────────────────────────────────

def get_projected_pt(cx, cy, w, h):
    horizon_y = int(h * 0.6)
    screen_y = horizon_y + int(cy * (h - horizon_y))
    perspective_scale = 0.3 + 0.7 * cy
    screen_x = int(w / 2 + cx * (w / 2) * perspective_scale)
    return screen_x, screen_y


def draw_court_background(w, h):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    horizon_y = int(h * 0.6)

    for y in range(horizon_y):
        t = y / max(horizon_y - 1, 1)
        color = tuple(
            int(SKY_GRADIENT_START[c] * (1 - t) + SKY_GRADIENT_END[c] * t)
            for c in range(3)
        )
        bg[y, :] = color

    court_top = np.array(COURT_GREEN, dtype=np.float32)
    court_bot = np.array(COURT_GREEN, dtype=np.float32) * 0.75
    for y in range(horizon_y, h):
        t = (y - horizon_y) / max(h - horizon_y - 1, 1)
        color = tuple(int(court_top[c] * (1 - t) + court_bot[c] * t) for c in range(3))
        bg[y, :] = color

    line_color = (200, 200, 200)
    line_thick = max(1, 2 * SSAA // 2)

    cv2.line(bg, get_projected_pt(-0.9, 0.95, w, h), get_projected_pt(0.9, 0.95, w, h), line_color, line_thick, cv2.LINE_AA)
    cv2.line(bg, get_projected_pt(-0.9, 0.55, w, h), get_projected_pt(0.9, 0.55, w, h), line_color, line_thick, cv2.LINE_AA)
    cv2.line(bg, get_projected_pt(0.0, 0.0, w, h),   get_projected_pt(0.0, 0.55, w, h), line_color, line_thick, cv2.LINE_AA)

    for x_norm in [-0.7, 0.7]:
        cv2.line(bg, get_projected_pt(x_norm, 0.0, w, h), get_projected_pt(x_norm, 0.95, w, h), line_color, line_thick, cv2.LINE_AA)
    for x_norm in [-0.9, 0.9]:
        cv2.line(bg, get_projected_pt(x_norm, 0.0, w, h), get_projected_pt(x_norm, 0.95, w, h), line_color, line_thick, cv2.LINE_AA)

    return bg


# ─────────────────────────────────────────
# Step 3: Stickman Drawing
# ─────────────────────────────────────────

def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    cx, cy = w / 2, h / 2
    x = cx + (lm.x * w - cx) * SCALE_FACTOR + OFFSET_X * SSAA
    y = cy + (lm.y * h - cy) * SCALE_FACTOR + OFFSET_Y * SSAA
    return (int(x), int(y))


def vis_ok(landmarks, *indices):
    """주어진 랜드마크 인덱스 모두 visibility >= VISIBILITY_THRESHOLD 이면 True."""
    return all(getattr(landmarks[i], 'visibility', 1.0) >= VISIBILITY_THRESHOLD for i in indices)


def draw_head(canvas, cx, cy, radius):
    cv2.circle(canvas, (cx, cy), radius, HEAD_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius, OUTLINE_COLOR, OUTLINE_THICKNESS, cv2.LINE_AA)


def draw_body_line(canvas, p1, p2, thickness=None):
    cv2.line(canvas, p1, p2, BODY_COLOR, thickness or BODY_THICKNESS, cv2.LINE_AA)


def draw_pentagon_torso(canvas, neck, l_shoulder, r_shoulder, l_hip, r_hip):
    pts = np.array([neck, r_shoulder, r_hip, l_hip, l_shoulder], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], BODY_COLOR, cv2.LINE_AA)
    cv2.polylines(canvas, [pts], isClosed=True, color=OUTLINE_COLOR,
                  thickness=OUTLINE_THICKNESS, lineType=cv2.LINE_AA)


def draw_hand(canvas, wrist, radius):
    cv2.circle(canvas, wrist, radius, HAND_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, wrist, radius, HAND_OUTLINE_COLOR, max(2 * SSAA, 2), cv2.LINE_AA)


def draw_shoe(canvas, ankle, heel, toe):
    sole_vec = np.array(toe) - np.array(heel)
    L = np.linalg.norm(sole_vec) + 1e-6
    u_sole = sole_vec / L
    perp = np.array([-u_sole[1], u_sole[0]])

    shoe_length = L * 1.4
    shoe_height = L * 0.55
    base_center = (np.array(heel, dtype=np.float64) + np.array(toe, dtype=np.float64)) / 2.0

    pts = np.array([
        base_center - u_sole * shoe_length * 0.45,
        base_center + u_sole * shoe_length * 0.55,
        base_center + u_sole * shoe_length * 0.55 + perp * shoe_height * 0.4,
        base_center + perp * shoe_height,
        np.array(ankle, dtype=np.float64) + perp * shoe_height * 0.2,
        base_center - u_sole * shoe_length * 0.45 + perp * shoe_height * 0.7,
    ], dtype=np.int32)

    cv2.fillPoly(canvas, [pts], SHOE_FILL_COLOR, cv2.LINE_AA)
    cv2.polylines(canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                  thickness=max(3 * SSAA, 2), lineType=cv2.LINE_AA)


def draw_shadow(canvas, l_ankle, r_ankle):
    cx = (l_ankle[0] + r_ankle[0]) // 2
    cy = max(l_ankle[1], r_ankle[1]) + 6 * SSAA
    spread = max(abs(l_ankle[0] - r_ankle[0]), 40 * SSAA)
    overlay = canvas.copy()
    cv2.ellipse(overlay, (cx, cy), (int(spread * 0.7), max(int(spread * 0.1), 6 * SSAA)),
                0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)


def draw_racket(canvas, wrist, elbow, head_r):
    wx, wy = wrist
    ex, ey = elbow
    dx, dy = wx - ex, wy - ey
    arm_len = math.sqrt(dx * dx + dy * dy) + 1e-6
    nx, ny = dx / arm_len, dy / arm_len

    grip_length = int(head_r * 0.9)
    frame_rx = int(head_r * 1.0)
    frame_ry = int(head_r * 1.35)

    grip_end_x = int(wx + nx * grip_length)
    grip_end_y = int(wy + ny * grip_length)
    cv2.line(canvas, (wx, wy), (grip_end_x, grip_end_y),
             RACKET_GRIP_COLOR, max(int(5 * SSAA), 4), cv2.LINE_AA)

    head_cx = int(grip_end_x + nx * frame_ry)
    head_cy = int(grip_end_y + ny * frame_ry)
    angle = math.degrees(math.atan2(ny, nx))

    cv2.ellipse(canvas, (head_cx, head_cy), (frame_rx, frame_ry),
                angle, 0, 360, RACKET_FRAME_COLOR, max(int(4 * SSAA), 3), cv2.LINE_AA)

    string_thick = max(1, SSAA)
    perp_x, perp_y = -ny, nx
    for frac in [-0.3, 0.0, 0.3]:
        sx = int(head_cx + perp_x * frame_rx * frac * 0.8)
        sy = int(head_cy + perp_y * frame_rx * frac * 0.8)
        cv2.line(canvas,
                 (int(sx - nx * frame_ry * 0.6), int(sy - ny * frame_ry * 0.6)),
                 (int(sx + nx * frame_ry * 0.6), int(sy + ny * frame_ry * 0.6)),
                 RACKET_STRING_COLOR, string_thick, cv2.LINE_AA)
    for frac in [-0.3, 0.0, 0.3]:
        sx = int(head_cx + nx * frame_ry * frac * 0.8)
        sy = int(head_cy + ny * frame_ry * frac * 0.8)
        cv2.line(canvas,
                 (int(sx - perp_x * frame_rx * 0.6), int(sy - perp_y * frame_rx * 0.6)),
                 (int(sx + perp_x * frame_rx * 0.6), int(sy + perp_y * frame_rx * 0.6)),
                 RACKET_STRING_COLOR, string_thick, cv2.LINE_AA)


def draw_stickman(canvas, landmarks, w, h, is_right_handed=True):
    lm = landmarks
    nose       = get_point(lm, NOSE, w, h)
    l_shoulder = get_point(lm, L_SHOULDER, w, h)
    r_shoulder = get_point(lm, R_SHOULDER, w, h)
    l_elbow    = get_point(lm, L_ELBOW, w, h)
    r_elbow    = get_point(lm, R_ELBOW, w, h)
    l_wrist    = get_point(lm, L_WRIST, w, h)
    r_wrist    = get_point(lm, R_WRIST, w, h)
    l_hip      = get_point(lm, L_HIP, w, h)
    r_hip      = get_point(lm, R_HIP, w, h)
    l_knee     = get_point(lm, L_KNEE, w, h)
    r_knee     = get_point(lm, R_KNEE, w, h)
    l_ankle    = get_point(lm, L_ANKLE, w, h)
    r_ankle    = get_point(lm, R_ANKLE, w, h)
    l_heel     = get_point(lm, L_HEEL, w, h)
    r_heel     = get_point(lm, R_HEEL, w, h)
    l_foot_idx = get_point(lm, L_FOOT_INDEX, w, h)
    r_foot_idx = get_point(lm, R_FOOT_INDEX, w, h)

    neck    = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
    mid_hip = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)

    shoulder_width = math.sqrt(
        (l_shoulder[0] - r_shoulder[0]) ** 2 + (l_shoulder[1] - r_shoulder[1]) ** 2
    )
    head_r  = max(int(shoulder_width * 0.42), 12 * SSAA)
    head_cx = neck[0]
    head_cy = neck[1] - int(head_r * 1.05)
    hand_r  = max(int(head_r * 0.22), 4 * SSAA)

    # 그림자 — 발목 둘 다 보일 때만
    if vis_ok(lm, L_ANKLE, R_ANKLE):
        draw_shadow(canvas, l_ankle, r_ankle)

    # 왼쪽 다리
    if vis_ok(lm, L_HIP, L_KNEE):
        draw_body_line(canvas, l_hip, l_knee)
    if vis_ok(lm, L_KNEE, L_ANKLE):
        draw_body_line(canvas, l_knee, l_ankle)
    # 오른쪽 다리
    if vis_ok(lm, R_HIP, R_KNEE):
        draw_body_line(canvas, r_hip, r_knee)
    if vis_ok(lm, R_KNEE, R_ANKLE):
        draw_body_line(canvas, r_knee, r_ankle)

    # 신발 — 발목·뒤꿈치·발끝 모두 보일 때만
    if vis_ok(lm, L_ANKLE, L_HEEL, L_FOOT_INDEX):
        draw_shoe(canvas, l_ankle, l_heel, l_foot_idx)
    if vis_ok(lm, R_ANKLE, R_HEEL, R_FOOT_INDEX):
        draw_shoe(canvas, r_ankle, r_heel, r_foot_idx)

    draw_pentagon_torso(canvas, neck, l_shoulder, r_shoulder, l_hip, r_hip)

    # 왼팔
    if vis_ok(lm, L_SHOULDER, L_ELBOW):
        draw_body_line(canvas, l_shoulder, l_elbow)
    if vis_ok(lm, L_ELBOW, L_WRIST):
        draw_body_line(canvas, l_elbow, l_wrist)
    # 오른팔
    if vis_ok(lm, R_SHOULDER, R_ELBOW):
        draw_body_line(canvas, r_shoulder, r_elbow)
    if vis_ok(lm, R_ELBOW, R_WRIST):
        draw_body_line(canvas, r_elbow, r_wrist)

    if vis_ok(lm, L_WRIST):
        draw_hand(canvas, l_wrist, hand_r)
    if vis_ok(lm, R_WRIST):
        draw_hand(canvas, r_wrist, hand_r)

    if is_right_handed:
        if vis_ok(lm, R_WRIST, R_ELBOW):
            draw_racket(canvas, r_wrist, r_elbow, head_r)
    else:
        if vis_ok(lm, L_WRIST, L_ELBOW):
            draw_racket(canvas, l_wrist, l_elbow, head_r)
    draw_body_line(canvas, neck, (head_cx, head_cy + head_r), thickness=BODY_THICKNESS)
    draw_head(canvas, head_cx, head_cy, head_r)


# ─────────────────────────────────────────
# Step 4: Video Processing
# ─────────────────────────────────────────

def process_video(video_url, name, is_right_handed):
    input_path  = f"{name}_input.mp4"
    output_path = f"{name}_stickman.mp4"

    download_model()
    actual_input = download_video(video_url, input_path)
    if not actual_input:
        print("[✗] Cannot proceed without input video.")
        return

    cap = cv2.VideoCapture(actual_input)
    if not cap.isOpened():
        print(f"[✗] Cannot open video: {input_path}")
        return

    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    render_w, render_h = orig_w * SSAA, orig_h * SSAA

    print(f"[i] Video: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames")
    print(f"[i] Rendering at {render_w}x{render_h} (SSAA={SSAA}x)")

    court_bg = draw_court_background(render_w, render_h)

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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    smoothed  = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000.0 / fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        canvas = court_bg.copy()

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            raw = result.pose_landmarks[0]
            if smoothed is None:
                smoothed = [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z,
                            visibility=getattr(lm, 'visibility', 1.0)) for lm in raw]
            else:
                for i in range(len(raw)):
                    smoothed[i].x = EMA_ALPHA * raw[i].x + (1 - EMA_ALPHA) * smoothed[i].x
                    smoothed[i].y = EMA_ALPHA * raw[i].y + (1 - EMA_ALPHA) * smoothed[i].y
                    smoothed[i].z = EMA_ALPHA * raw[i].z + (1 - EMA_ALPHA) * smoothed[i].z
                    smoothed[i].visibility = getattr(raw[i], 'visibility', 1.0)
            draw_stickman(canvas, smoothed, render_w, render_h, is_right_handed)

        final = cv2.resize(canvas, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        out.write(final)

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"  Processing: frame {frame_idx}/{total_frames} ({pct:.1f}%)")

    cap.release()
    out.release()
    landmarker.close()

    print(f"\n[✓] Stickman video saved: {output_path}")
    print(f"    Resolution: {orig_w}x{orig_h} @ {fps:.1f}fps, {frame_idx} frames")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"  TENNIS STICKMAN ANIMATION GENERATOR v3")
    print(f"  동작: {args.name}  |  손: {'왼손' if args.left else '오른손'}")
    print("=" * 60)
    process_video(args.url, args.name, is_right_handed=not args.left)
