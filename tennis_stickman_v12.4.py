# 테니스 동작 영상을 스틱맨 애니메이션으로 변환하는 생성기 (v12.4: 오디오+랜드마크 이중검증 → 팔꿈치조건 폴백 통합 앙상블)
"""
Tennis Stickman Animation Generator v10 (TrackNet Ball Tracking)
사용법:
  python tennis_stickman_v9.py <YouTube_URL_또는_로컬파일> <동작명> [--left] [--speed <배속>] [--strobe] [--two-handed]

v9 변경 사항:
  1. 원본 오디오 자동 합성: 렌더링 완료 후 원본 영상의 오디오를 스틱맨 영상에 자동으로 합성
  2. 양손 백핸드 그립 (--two-handed): 왼손을 라켓 그립 상단에 고정하여 양손 백핸드 표현
  3. 와이퍼 물리 자동 비활성화: --two-handed 사용 시 lag_scale=0으로 강제 설정
  4. 재생 배속 버그 수정: 손목 속도 변수명 speed → wrist_vel 충돌 수정 (21분 영상 버그 해결)
  5. ffmpeg stdin 파이프 인코딩: cv2 mp4v FPS 메타데이터 버그 대체
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
        description="테니스 스틱맨 애니메이션 생성기 v8 (다중 잔상 및 와이퍼 스윙 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url",  help="YouTube 영상 URL 또는 로컬 파일 경로")
    parser.add_argument("name", help="동작명 (파일명에 사용, 예: federer_wiper)")
    parser.add_argument("--left", action="store_true", help="왼손잡이 선수 (기본: 오른손잡이)")
    parser.add_argument("--label", default=None, help="화면에 표시할 동작 이름 자막 (예: 포핸드)")
    parser.add_argument("--desc",  default=None, help="자막 아래 줄 설명 (선택)")
    parser.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율 (예: 0.5는 슬로우 모션)")
    parser.add_argument("--strobe", action="store_true", help="다중 잔상 효과(Stroboscopic Effect) 활성화")
    parser.add_argument("--strobe-frames", type=int, default=32, help="추적할 히스토리 프레임 수")
    parser.add_argument("--strobe-step", type=int, default=4, help="잔상 프레임 샘플링 간격")
    parser.add_argument("--lag-scale", type=float, default=0.0, help="손목 래그(Wrist Lag) 각도 오프셋 스케일 (0.0으로 설정 시 물리 비활성화)")
    parser.add_argument("--no-trail", action="store_true", help="네온 스윙 궤적(노란색/파란색 선) 표시 비활성화")
    parser.add_argument("--two-handed", action="store_true", help="양손 그립 모드: 왼손을 라켓 그립 위에 배치하고 와이퍼 물리 비활성화")
    parser.add_argument("--audio-impact", action="store_true", help="오디오 타구음 기반 임팩트 감지 (가장 정확)")
    parser.add_argument("--ball-track", action="store_true", help="TrackNet 딥러닝 공 추적 기반 임팩트 감지")
    parser.add_argument("--tracknet-model", type=str,
                        default=r"C:\TrackNet\model_best.pt",
                        help="TrackNet 가중치 파일 경로 (기본: Desktop/model_best.pt)")
    parser.add_argument("--impact-frame", type=int, nargs="+", default=None,
                        metavar="N", help="임팩트 프레임 번호 직접 지정 (예: --impact-frame 690 또는 여러 개: 300 690)")
    parser.add_argument("--ensemble", action="store_true",
                        help="앙상블 임팩트 감지: 오디오+팔신장 동시 분석 후 합의 프레임 선택")
    return parser.parse_args()


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = r"C:\Users\박범서\Downloads\stickman\pose_landmarker_full.task"

SSAA  = 2      # 슈퍼샘플링 배수 (렌더 = 출력 × SSAA)
OUT_H = 720    # 출력 세로 해상도
LW = SSAA

# 전역 손목 각도 및 라켓 상태 캐시 (프레임 간 스무딩)
_racket_offset_prev = None
_racket_face_prev = None

def configure_thickness(render_h):
    """렌더 높이에 맞춰 선 두께 단위와 명명된 두께 상수를 설정."""
    global LW, LIMB_THICKNESS, NECK_THICKNESS, HEAD_OUTLINE_THICKNESS, TORSO_OUTLINE_THICKNESS
    LW = SSAA * (render_h / 720.0)
    LIMB_THICKNESS          = max(int(6 * LW), 2)
    NECK_THICKNESS          = max(int(7 * LW), 2)
    HEAD_OUTLINE_THICKNESS  = max(int(8 * LW), 2)
    TORSO_OUTLINE_THICKNESS = max(int(6 * LW), 2)

BODY_COLOR        = (20, 20, 20)
LIMB_THICKNESS    = 6 * SSAA
NECK_THICKNESS    = 7 * SSAA
HEAD_OUTLINE_THICKNESS  = 8 * SSAA
TORSO_OUTLINE_THICKNESS = 6 * SSAA
HEAD_FILL_COLOR   = (255, 255, 255)
OUTLINE_COLOR     = (20, 20, 20)
SHOE_FILL_COLOR   = (155, 155, 155)
SHOE_OUTLINE_COLOR = (20, 20, 20)
HAND_FILL_COLOR   = (60, 60, 60)
HAND_OUTLINE_COLOR = (20, 20, 20)

RACKET_FRAME_COLOR  = (30, 30, 220)
RACKET_STRING_COLOR = (210, 210, 215)
RACKET_GRIP_COLOR   = (40, 40, 40)

COURT_GREEN         = (78, 115, 76)
SKY_GRADIENT_START  = (215, 215, 215)
SKY_GRADIENT_END    = (238, 238, 238)

ONE_EURO_MIN_CUTOFF = 0.2
ONE_EURO_BETA       = 0.02
ONE_EURO_D_CUTOFF   = 1.0

SCALE_FACTOR = 0.9
OFFSET_X     = 0
OFFSET_Y     = 10

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"  # 한글 자막용

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
# One-Euro 필터 (모션 스무딩)
# ─────────────────────────────────────────

class OneEuroFilter:
    def __init__(self, freq, min_cutoff, beta, d_cutoff):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        edx = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = edx
        return x_hat


class PoseSmoother:
    def __init__(self, freq, n=33):
        mk = lambda: OneEuroFilter(freq, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.fx = [mk() for _ in range(n)]
        self.fy = [mk() for _ in range(n)]
        self.fz = [mk() for _ in range(n)]

    def apply(self, raw):
        return [SimpleNamespace(
            x=self.fx[i](raw[i].x),
            y=self.fy[i](raw[i].y),
            z=self.fz[i](raw[i].z),
        ) for i in range(len(raw))]


# ─────────────────────────────────────────
# 텍스트/자막 레이어
# ─────────────────────────────────────────

def draw_label(frame_bgr, label, desc=None):
    from PIL import Image, ImageDraw, ImageFont
    h, w = frame_bgr.shape[:2]
    pad = max(int(h * 0.025), 8)
    f_label = ImageFont.truetype(FONT_PATH, max(int(h / 13), 18))
    f_desc  = ImageFont.truetype(FONT_PATH, max(int(h / 26), 12)) if desc else None

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")

    lb = draw.textbbox((0, 0), label, font=f_label)
    lw_, lh_ = lb[2] - lb[0], lb[3] - lb[1]
    dw_ = dh_ = 0
    if desc:
        db = draw.textbbox((0, 0), desc, font=f_desc)
        dw_, dh_ = db[2] - db[0], db[3] - db[1]

    bar_w = max(lw_, dw_) + pad * 2
    bar_h = lh_ + (dh_ + pad // 2 if desc else 0) + pad * 2
    x0, y0 = pad, pad
    draw.rounded_rectangle([x0, y0, x0 + bar_w, y0 + bar_h],
                           radius=pad, fill=(20, 20, 20, 150))
    draw.text((x0 + pad, y0 + pad - lb[1]), label, font=f_label, fill=(255, 255, 255, 255))
    if desc:
        draw.text((x0 + pad, y0 + pad + lh_ + pad // 2 - db[1]), desc,
                  font=f_desc, fill=(210, 210, 210, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────
# 다운로드 및 파일 핸들링
# ─────────────────────────────────────────

def download_video(video_url, input_path):
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
# 코트 배경 그리기
# ─────────────────────────────────────────

def get_projected_pt(cx, cy, w, h):
    horizon_y = int(h * 0.65)
    screen_y = horizon_y + int(cy * (h - horizon_y))
    perspective_scale = 0.3 + 0.7 * cy
    screen_x = int(w / 2 + cx * (w / 2) * perspective_scale)
    return screen_x, screen_y


def draw_court_background(w, h):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    horizon_y = int(h * 0.65)

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
    line_thick = max(1, int(round(LW)))

    cv2.line(bg, get_projected_pt(-0.9, 0.95, w, h), get_projected_pt(0.9, 0.95, w, h), line_color, line_thick, cv2.LINE_AA)
    cv2.line(bg, get_projected_pt(-0.9, 0.55, w, h), get_projected_pt(0.9, 0.55, w, h), line_color, line_thick, cv2.LINE_AA)
    cv2.line(bg, get_projected_pt(0.0, 0.0, w, h),   get_projected_pt(0.0, 0.55, w, h), line_color, line_thick, cv2.LINE_AA)

    for x_norm in [-0.7, 0.7]:
        cv2.line(bg, get_projected_pt(x_norm, 0.0, w, h), get_projected_pt(x_norm, 0.95, w, h), line_color, line_thick, cv2.LINE_AA)
    for x_norm in [-0.9, 0.9]:
        cv2.line(bg, get_projected_pt(x_norm, 0.0, w, h), get_projected_pt(x_norm, 0.95, w, h), line_color, line_thick, cv2.LINE_AA)

    return bg


# ─────────────────────────────────────────
# 스틱맨 부위별 렌더링 함수
# ─────────────────────────────────────────

def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    cx, cy = w / 2, h / 2
    x = cx + (lm.x * w - cx) * SCALE_FACTOR + OFFSET_X * LW
    y = cy + (lm.y * h - cy) * SCALE_FACTOR + OFFSET_Y * LW
    return (int(x), int(y))


def draw_head(canvas, cx, cy, radius):
    cv2.circle(canvas, (cx, cy), radius, HEAD_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius, OUTLINE_COLOR, HEAD_OUTLINE_THICKNESS, cv2.LINE_AA)


def draw_body_line(canvas, p1, p2, thickness=None):
    cv2.line(canvas, p1, p2, BODY_COLOR, thickness or LIMB_THICKNESS, cv2.LINE_AA)


def draw_pentagon_torso(canvas, neck, l_shoulder, r_shoulder, l_hip, r_hip):
    pts = np.array([neck, r_shoulder, r_hip, l_hip, l_shoulder], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(canvas, [pts], isClosed=True, color=OUTLINE_COLOR,
                  thickness=TORSO_OUTLINE_THICKNESS, lineType=cv2.LINE_AA)


def draw_hand(canvas, wrist, radius):
    cv2.circle(canvas, wrist, radius, HAND_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, wrist, radius, HAND_OUTLINE_COLOR, max(int(3 * LW), 3), cv2.LINE_AA)


# 전역 신발 상태 캐시 (왼발/오른발 구분)
_shoe_blend_cache = {"left": None, "right": None}

def draw_shoe(canvas, ankle, heel, toe, knee, size, is_back_view, side_key="left"):
    global _shoe_blend_cache

    # 왼발(밝은 회색)·오른발(어두운 회색)로 구분
    shoe_fill = (210, 210, 210) if side_key == "left" else (90, 90, 90)

    ankle = np.array(ankle, dtype=np.float64)
    heel  = np.array(heel,  dtype=np.float64)
    toe   = np.array(toe,   dtype=np.float64)
    knee  = np.array(knee,  dtype=np.float64)

    foot = toe - heel
    L = np.linalg.norm(foot)

    # 튜닝 가이드 조합: 더 부드러운 전환 세팅 (구간 넓히기)
    threshold_low  = size * 0.20
    threshold_high = size * 0.65
    alpha_temporal = 0.30

    # 0.0(정면)과 1.0(측면) 사이의 raw 블렌딩 비율 계산
    if L <= threshold_low:
        blend_raw = 0.0
    elif L >= threshold_high:
        blend_raw = 1.0
    else:
        blend_raw = (L - threshold_low) / (threshold_high - threshold_low)

    # 시간적 스무딩(Temporal Smoothing) 적용
    prev_blend = _shoe_blend_cache.get(side_key)
    if prev_blend is None:
        blend_val = blend_raw
    else:
        blend_val = prev_blend + alpha_temporal * (blend_raw - prev_blend)
    _shoe_blend_cache[side_key] = blend_val

    # 헬퍼 렌더러 정의
    def render_front_view(target_canvas):
        v = ankle - knee
        d_dir = v / (np.linalg.norm(v) + 1e-6)
        w_dir = np.array([-d_dir[1], d_dir[0]])

        if is_back_view:
            profile = [
                (-0.22, 0.00), (-0.32, 0.20), (-0.40, 0.60), (-0.20, 0.65),
                ( 0.20, 0.65), ( 0.40, 0.60), ( 0.32, 0.20), ( 0.22, 0.00),
            ]
            pts = np.array([
                (ankle + w_dir * fx * size + d_dir * fy * size)
                for fx, fy in profile
            ], dtype=np.int32)

            cv2.fillPoly(target_canvas, [pts], shoe_fill, cv2.LINE_AA)
            cv2.polylines(target_canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                          thickness=max(int(3 * LW), 2), lineType=cv2.LINE_AA)

            sole = np.array([
                (ankle - w_dir * (size * 0.40) + d_dir * (size * 0.60)),
                (ankle - w_dir * (size * 0.20) + d_dir * (size * 0.65)),
                (ankle + w_dir * (size * 0.20) + d_dir * (size * 0.65)),
                (ankle + w_dir * (size * 0.40) + d_dir * (size * 0.60))
            ], dtype=np.int32)
            cv2.polylines(target_canvas, [sole], isClosed=False, color=(110, 110, 110),
                          thickness=max(int(2 * LW), 2), lineType=cv2.LINE_AA)

            heel_strip_start = (ankle + d_dir * (size * 0.05)).astype(np.int32)
            heel_strip_end = (ankle + d_dir * (size * 0.22)).astype(np.int32)
            cv2.line(target_canvas, heel_strip_start, heel_strip_end, SHOE_OUTLINE_COLOR,
                     max(int(2.5 * LW), 2), cv2.LINE_AA)
        else:
            profile = [
                (-0.22, 0.00), (-0.35, 0.25), (-0.42, 0.70), (-0.20, 0.76),
                ( 0.20, 0.76), ( 0.42, 0.70), ( 0.35, 0.25), ( 0.22, 0.00),
            ]
            pts = np.array([
                (ankle + w_dir * fx * size + d_dir * fy * size)
                for fx, fy in profile
            ], dtype=np.int32)

            cv2.fillPoly(target_canvas, [pts], shoe_fill, cv2.LINE_AA)
            cv2.polylines(target_canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                          thickness=max(int(3 * LW), 2), lineType=cv2.LINE_AA)

            sole = np.array([
                (ankle - w_dir * (size * 0.42) + d_dir * (size * 0.70)),
                (ankle - w_dir * (size * 0.20) + d_dir * (size * 0.76)),
                (ankle + w_dir * (size * 0.20) + d_dir * (size * 0.76)),
                (ankle + w_dir * (size * 0.42) + d_dir * (size * 0.70))
            ], dtype=np.int32)
            cv2.polylines(target_canvas, [sole], isClosed=False, color=(110, 110, 110),
                          thickness=max(int(2 * LW), 2), lineType=cv2.LINE_AA)

            lace_start = (ankle + d_dir * (size * 0.12))
            lace_end = (ankle + d_dir * (size * 0.45))
            cv2.line(target_canvas, lace_start.astype(np.int32), lace_end.astype(np.int32),
                     SHOE_OUTLINE_COLOR, max(int(1 * LW), 1), cv2.LINE_AA)

            for frac in [0.20, 0.30, 0.40]:
                bar_center = ankle + d_dir * (size * frac)
                bar_left = (bar_center - w_dir * (size * 0.12)).astype(np.int32)
                bar_right = (bar_center + w_dir * (size * 0.12)).astype(np.int32)
                cv2.line(target_canvas, bar_left, bar_right, (255, 255, 255),
                         max(int(1 * LW), 1), cv2.LINE_AA)

            toe_cap_center = ankle + d_dir * (size * 0.52)
            toe_cap_left = (toe_cap_center - w_dir * (size * 0.32)).astype(np.int32)
            toe_cap_right = (toe_cap_center + w_dir * (size * 0.32)).astype(np.int32)
            cv2.line(target_canvas, toe_cap_left, toe_cap_right, SHOE_OUTLINE_COLOR,
                     max(int(1.2 * LW), 2), cv2.LINE_AA)

    def render_side_view(target_canvas):
        u = foot / L
        perp = np.array([-u[1], u[0]])
        if perp[1] < 0:
            perp = -perp

        length = size * 1.15
        height = size * 0.62

        profile = [
            (-0.12, 0.00), (-0.22, 0.20), (-0.24, 0.50), (-0.18, 0.78),
            (-0.06, 0.94), ( 0.18, 1.00), ( 0.48, 1.00), ( 0.72, 0.97),
            ( 0.90, 0.88), ( 1.00, 0.72), ( 1.02, 0.54), ( 0.96, 0.36),
            ( 0.82, 0.24), ( 0.60, 0.16), ( 0.38, 0.11), ( 0.18, 0.06),
        ]
        pts = np.array([
            (ankle + u * fx * length + perp * fy * height)
            for fx, fy in profile
        ], dtype=np.int32)

        cv2.fillPoly(target_canvas, [pts], SHOE_FILL_COLOR, cv2.LINE_AA)
        cv2.polylines(target_canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                      thickness=max(int(3 * LW), 2), lineType=cv2.LINE_AA)

        sole = np.array([
            (ankle + u * fx * length + perp * fy * height)
            for fx, fy in [(-0.06, 0.94), (0.18, 1.00), (0.48, 1.00), (0.72, 0.97)]
        ], dtype=np.int32)
        cv2.polylines(target_canvas, [sole], isClosed=False, color=(110, 110, 110),
                      thickness=max(int(2 * LW), 2), lineType=cv2.LINE_AA)

    # 블렌딩 렌더링 적용 (임시 캔버스 활용)
    if blend_val <= 0.001:
        render_front_view(canvas)
    elif blend_val >= 0.999:
        render_side_view(canvas)
    else:
        canvas_front = canvas.copy()
        canvas_side = canvas.copy()
        render_front_view(canvas_front)
        render_side_view(canvas_side)
        cv2.addWeighted(canvas_front, 1.0 - blend_val, canvas_side, blend_val, 0, dst=canvas)


def draw_shadow(canvas, l_ankle, r_ankle):
    cx = (l_ankle[0] + r_ankle[0]) // 2
    cy = max(l_ankle[1], r_ankle[1]) + int(6 * LW)
    spread = max(abs(l_ankle[0] - r_ankle[0]), int(40 * LW))
    overlay = canvas.copy()
    cv2.ellipse(overlay, (cx, cy), (int(spread * 0.7), max(int(spread * 0.1), int(6 * LW))),
                0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)


def draw_shorts(canvas, l_hip, r_hip, l_knee, r_knee, head_r):
    """
    PGNC 반바지 실제 외곽선 실루엣 기반 렌더링 (v11)
    - 정규화 좌표: 반바지 이미지에서 추출한 10-point 외곽선
    - 허리선(l_hip~r_hip) + 밑단(무릎 45% 지점) 기준으로 변환
    """
    l_hip_arr   = np.array(l_hip,   dtype=np.float64)
    r_hip_arr   = np.array(r_hip,   dtype=np.float64)
    l_knee_arr  = np.array(l_knee,  dtype=np.float64)
    r_knee_arr  = np.array(r_knee,  dtype=np.float64)

    # 반바지 밑단: 엉덩이~무릎의 45% 지점
    l_bottom = l_hip_arr + 0.45 * (l_knee_arr - l_hip_arr)
    r_bottom = r_hip_arr + 0.45 * (r_knee_arr - r_hip_arr)

    # 반바지 박스의 4개 꼭짓점 정의
    # 왼쪽 허리 → 오른쪽 허리 → 오른쪽 밑단 → 왼쪽 밑단
    # 정규화 좌표계: x(0=왼쪽, 1=오른쪽), y(0=허리, 1=밑단)
    # 이미지에서 추출한 10-point 외곽선 (PGNC 반바지)
    # 순서: 좌상단 → 좌하단 → 가랑이 → 우하단 → 우상단 → 허리 중앙
    SHORTS_NORM_PTS = np.array([
        [0.2029, 0.003 ],  # 0: 왼쪽 허리선 안쪽
        [0.0692, 0.4012],  # 1: 왼쪽 옆선 중간
        [0.0048, 0.8743],  # 2: 왼쪽 밑단 끝
        [0.4511, 0.997 ],  # 3: 왼쪽 가랑이 밑단
        [0.5036, 0.7904],  # 4: 가랑이 중앙 (오목)
        [0.5561, 0.988 ],  # 5: 오른쪽 가랑이 밑단
        [0.9976, 0.8743],  # 6: 오른쪽 밑단 끝
        [0.9379, 0.4521],  # 7: 오른쪽 옆선 중간
        [0.7947, 0.0   ],  # 8: 오른쪽 허리선 안쪽
        [0.5012, 0.0778],  # 9: 허리 중앙 (고무밴드)
    ], dtype=np.float64)

    # 좌표 변환: 정규화(0~1) → 스크린 픽셀
    # x축: l_hip(x=0) ~ r_hip(x=1) 방향 벡터
    # y축: hip(y=0) ~ bottom(y=1) 방향 벡터
    hip_vec   = r_hip_arr - l_hip_arr          # 허리 방향 벡터 (x축)
    # 왼/오른 각각 다리 방향이 다를 수 있어 평균 사용
    l_leg_vec = l_bottom - l_hip_arr
    r_leg_vec = r_bottom - r_hip_arr

    pts_screen = []
    for nx, ny in SHORTS_NORM_PTS:
        # 허리선 위의 점: l_hip + nx * (r_hip - l_hip)
        waist_pt = l_hip_arr + nx * hip_vec
        # 다리 방향: 왼쪽(nx<0.5)은 l_leg_vec, 오른쪽은 r_leg_vec 가중 블렌딩
        leg_vec  = (1.0 - nx) * l_leg_vec + nx * r_leg_vec
        # 최종 스크린 좌표
        pt = waist_pt + ny * leg_vec
        pts_screen.append(pt)

    pts_screen = np.array(pts_screen, dtype=np.int32)

    # 1. 반바지 채우기 (차콜 컬러: PGNC 반바지 색상)
    shorts_color = (45, 45, 45)
    cv2.fillPoly(canvas, [pts_screen], shorts_color, cv2.LINE_AA)

    # 2. 허리 밴드 (약간 밝은 선)
    waist_left  = pts_screen[0]
    waist_right = pts_screen[8]
    waist_mid   = pts_screen[9]
    band_color  = (70, 70, 70)
    band_thick  = max(int(LW * 1.5), 2)
    cv2.line(canvas, tuple(waist_left), tuple(waist_mid),   band_color, band_thick, cv2.LINE_AA)
    cv2.line(canvas, tuple(waist_mid),  tuple(waist_right), band_color, band_thick, cv2.LINE_AA)

    # 3. 가운데 주름선 (가랑이 위 → 허리 중앙)
    crease_top    = pts_screen[9]                      # 허리 중앙
    crease_bottom = pts_screen[4]                      # 가랑이 오목점
    crease_color  = (75, 75, 75)
    crease_thick  = max(1, int(round(LW * 0.8)))
    cv2.line(canvas, tuple(crease_top), tuple(crease_bottom), crease_color, crease_thick, cv2.LINE_AA)

    # 4. 외곽선
    cv2.polylines(canvas, [pts_screen], isClosed=True,
                  color=OUTLINE_COLOR, thickness=TORSO_OUTLINE_THICKNESS, lineType=cv2.LINE_AA)


def draw_racket(canvas, wrist, elbow, head_r, nx_racket=None, ny_racket=None, racket_face_ratio=1.0):
    wx, wy = wrist
    ex, ey = elbow
    
    if nx_racket is None or ny_racket is None:
        dx, dy = wx - ex, wy - ey
        arm_len = math.sqrt(dx * dx + dy * dy) + 1e-6
        nx_racket, ny_racket = dx / arm_len, dy / arm_len

    grip_length = int(head_r * 1.2)
    frame_rx = int(head_r * 1.15 * racket_face_ratio)
    frame_ry = int(head_r * 1.5)

    grip_end_x = int(wx + nx_racket * grip_length)
    grip_end_y = int(wy + ny_racket * grip_length)
    cv2.line(canvas, (wx, wy), (grip_end_x, grip_end_y),
             RACKET_GRIP_COLOR, max(int(8 * LW), 6), cv2.LINE_AA)

    head_cx = int(grip_end_x + nx_racket * frame_ry)
    head_cy = int(grip_end_y + ny_racket * frame_ry)
    angle = math.degrees(math.atan2(ny_racket, nx_racket))

    # 90도 회전 버그 수정: frame_ry를 장축, frame_rx를 단축으로 대입
    cv2.ellipse(canvas, (head_cx, head_cy), (frame_ry, frame_rx),
                angle, 0, 360, RACKET_FRAME_COLOR, max(int(7 * LW), 5), cv2.LINE_AA)

    string_thick = max(1, int(round(LW)))
    perp_x, perp_y = -ny_racket, nx_racket
    for frac in [-0.3, 0.0, 0.3]:
        sx = int(head_cx + perp_x * frame_rx * frac * 0.8)
        sy = int(head_cy + perp_y * frame_rx * frac * 0.8)
        cv2.line(canvas,
                 (int(sx - nx_racket * frame_ry * 0.6), int(sy - ny_racket * frame_ry * 0.6)),
                 (int(sx + nx_racket * frame_ry * 0.6), int(sy + ny_racket * frame_ry * 0.6)),
                 RACKET_STRING_COLOR, string_thick, cv2.LINE_AA)
    for frac in [-0.3, 0.0, 0.3]:
        sx = int(head_cx + nx_racket * frame_ry * frac * 0.8)
        sy = int(head_cy + ny_racket * frame_ry * frac * 0.8)
        cv2.line(canvas,
                 (int(sx - perp_x * frame_rx * 0.6), int(sy - perp_y * frame_rx * 0.6)),
                 (int(sx + perp_x * frame_rx * 0.6), int(sy + perp_y * frame_rx * 0.6)),
                 RACKET_STRING_COLOR, string_thick, cv2.LINE_AA)


# ─────────────────────────────────────────
# 생체역학 및 잔상 지원 렌더러
# ─────────────────────────────────────────

class PhaseSmoother:
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.history = []
        
    def add_and_get(self, phase):
        self.history.append(phase)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return max(set(self.history), key=self.history.count)


def calculate_angle_2d(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def draw_angle_overlay(canvas, joint, p1, p2, color):
    angle_val = calculate_angle_2d(p1, joint, p2)
    v1 = np.array(p1) - np.array(joint)
    v2 = np.array(p2) - np.array(joint)
    ang1 = math.atan2(v1[1], v1[0])
    ang2 = math.atan2(v2[1], v2[0])
    deg1 = int(np.degrees(ang1))
    deg2 = int(np.degrees(ang2))
    
    diff = (deg2 - deg1) % 360
    if diff > 180:
        start_angle = deg2
        end_angle = deg1 + 360
    else:
        start_angle = deg1
        end_angle = deg2
        
    overlay = canvas.copy()
    radius = int(13 * LW)
    
    cv2.ellipse(overlay, joint, (radius, radius), 0, start_angle, end_angle, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
    
    cv2.ellipse(canvas, joint, (radius, radius), 0, start_angle, end_angle, color, max(1, int(1.2 * LW)), cv2.LINE_AA)
    
    bisector_deg = (start_angle + end_angle) / 2
    bisector_rad = np.radians(bisector_deg)
    
    text_dist = radius + int(8 * LW)
    tx = int(joint[0] + text_dist * np.cos(bisector_rad))
    ty = int(joint[1] + text_dist * np.sin(bisector_rad))
    
    text = f"{int(round(angle_val))}°"
    rgb_color = (color[2], color[1], color[0])
    return (text, (tx, ty), rgb_color)


def draw_texts_pil(canvas, tasks, font_size):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype(FONT_PATH, int(round(font_size)))
    
    for text, (x, y), color in tasks:
        tb = draw.textbbox((0, 0), text, font=font)
        outline_color = (20, 20, 20, 255)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy - tb[1]), text, font=font, fill=outline_color)
        draw.text((x, y - tb[1]), text, font=font, fill=color + (255,))
        
    canvas_new = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    np.copyto(canvas, canvas_new)


def draw_glowing_trail(canvas, trail, color):
    if len(trail) < 2:
        return
    n = len(trail)
    overlay = canvas.copy()
    for i in range(1, n):
        t = i / (n - 1)
        thick = max(1, int(10 * LW * t))
        cv2.line(overlay, trail[i-1], trail[i], color, thick, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    
    overlay_core = canvas.copy()
    for i in range(1, n):
        t = i / (n - 1)
        thick = max(1, int(3 * LW * t))
        cv2.line(overlay_core, trail[i-1], trail[i], (255, 255, 255), thick, cv2.LINE_AA)
    cv2.addWeighted(overlay_core, 0.65, canvas, 0.35, 0, canvas)


def detect_serve_phase(lm, is_right_handed, elbow_angle):
    h_wrist = lm[R_WRIST] if is_right_handed else lm[L_WRIST]
    h_shoulder = lm[R_SHOULDER] if is_right_handed else lm[L_SHOULDER]
    h_elbow = lm[R_ELBOW] if is_right_handed else lm[L_ELBOW]
    nh_wrist = lm[L_WRIST] if is_right_handed else lm[R_WRIST]
    nh_shoulder = lm[L_SHOULDER] if is_right_handed else lm[R_SHOULDER]
    nose = lm[NOSE]
    
    if h_wrist.y < nose.y and elbow_angle > 155:
        return "Impact"
    if h_elbow.y < h_shoulder.y + 0.05 and elbow_angle < 95 and h_wrist.y > h_elbow.y:
        return "Racket Drop"
    if nh_wrist.y < nh_shoulder.y - 0.05 and elbow_angle < 130 and elbow_angle > 60:
        return "Trophy Pose"
    if is_right_handed:
        if h_wrist.y > h_shoulder.y and h_wrist.x < lm[L_SHOULDER].x:
            return "Follow Through"
    else:
        if h_wrist.y > h_shoulder.y and h_wrist.x > lm[R_SHOULDER].x:
            return "Follow Through"
    return "Preparation"


def detect_groundstroke_phase(lm, is_right_handed, elbow_angle):
    h_wrist = lm[R_WRIST] if is_right_handed else lm[L_WRIST]
    h_shoulder = lm[R_SHOULDER] if is_right_handed else lm[L_SHOULDER]
    nh_shoulder = lm[L_SHOULDER] if is_right_handed else lm[R_SHOULDER]
    h_hip = lm[R_HIP] if is_right_handed else lm[L_HIP]
    
    def dist_2d(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
    d_opp_shoulder = dist_2d(h_wrist, nh_shoulder)
    
    if d_opp_shoulder < 0.18 and h_wrist.y < h_shoulder.y + 0.05:
        return "Follow Through"
    if elbow_angle > 135 and h_wrist.y > h_shoulder.y - 0.05 and h_wrist.y < h_hip.y + 0.1:
        return "Impact"
    if d_opp_shoulder > 0.35:
        return "Take Back"
    return "Preparation"


def draw_phase_badge(canvas, phase_name, w, h):
    if not phase_name:
        return canvas
    from PIL import Image, ImageDraw, ImageFont
    pad = max(int(h * 0.02), 6)
    font = ImageFont.truetype(FONT_PATH, max(int(h / 24), 14))
    
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    
    text = f"Phase: {phase_name}"
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    
    x1 = w - tw - pad * 3
    y1 = pad
    x2 = w - pad
    y2 = y1 + th + pad * 2
    
    border_color = (200, 200, 200, 255)
    if "Trophy" in phase_name:
        border_color = (255, 165, 0, 255)
    elif "Drop" in phase_name or "Take" in phase_name:
        border_color = (230, 50, 255, 255)
    elif "Impact" in phase_name:
        border_color = (50, 255, 50, 255)
    elif "Follow" in phase_name:
        border_color = (50, 180, 255, 255)
    elif "Preparation" in phase_name:
        border_color = (180, 180, 180, 255)
        
    draw.rounded_rectangle([x1, y1, x2, y2], radius=pad, fill=(20, 20, 20, 180),
                           outline=border_color, width=max(1, int(1.5 * LW)))
    draw.text((x1 + pad * 1.5, y1 + pad - tb[1]), text, font=font, fill=(255, 255, 255, 255))
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────
# 잔상 그리기 헬퍼 함수
# ─────────────────────────────────────────

def draw_ghost_figure(canvas, state, is_right_handed):
    # 각 관절 좌표 추출
    l_shoulder = state["l_shoulder"]
    r_shoulder = state["r_shoulder"]
    l_elbow    = state["l_elbow"]
    r_elbow    = state["r_elbow"]
    l_wrist    = state["l_wrist"]
    r_wrist    = state["r_wrist"]
    
    nx_racket  = state["nx_racket"]
    ny_racket  = state["ny_racket"]
    racket_face_ratio = state["racket_face_ratio"]
    head_r     = state["head_r"]
    hand_r     = state["hand_r"]
    
    joint_r = max(LIMB_THICKNESS // 2, int(2 * LW))
    
    # 오른손잡이이면 오른팔과 라켓, 왼손잡이이면 왼팔과 라켓만 잔상으로 그림
    h_shoulder = r_shoulder if is_right_handed else l_shoulder
    h_elbow = r_elbow if is_right_handed else l_elbow
    h_wrist = r_wrist if is_right_handed else l_wrist
    
    # 팔 그리기 (어깨-팔꿈치-손목)
    draw_body_line(canvas, h_shoulder, h_elbow)
    cv2.circle(canvas, h_elbow, joint_r, BODY_COLOR, -1, cv2.LINE_AA)
    draw_body_line(canvas, h_elbow, h_wrist)
    draw_hand(canvas, h_wrist, hand_r)
    
    # 라켓 그리기
    draw_racket(canvas, h_wrist, h_elbow, head_r, nx_racket, ny_racket, racket_face_ratio)


# ─────────────────────────────────────────
# 메인 스틱맨 그리기 함수
# ─────────────────────────────────────────

def draw_stickman(canvas, landmarks, w, h, is_right_handed=True, racket_trail=None, hand_trail=None,
                  phase_smoother=None, is_serve=True, k=9999, strobe_history=None, strobe_frames=32, strobe_step=4, lag_scale=1.0, two_handed=False):
    lm = landmarks
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
    head_r  = max(int(shoulder_width * 0.55), int(16 * LW))
    hand_r  = max(int(head_r * 0.26), int(6 * LW))

    neck_v = np.array(neck, dtype=np.float64)
    spine  = neck_v - np.array(mid_hip, dtype=np.float64)
    sn = np.linalg.norm(spine)
    up = spine / sn if sn > 1e-3 else np.array([0.0, -1.0])
    neck_len = int(head_r * 0.5)
    head_center = neck_v + up * (neck_len + head_r)
    head_cx, head_cy = int(head_center[0]), int(head_center[1])
    head_bottom = (head_center - up * head_r).astype(int)

    is_back_view = lm[R_SHOULDER].x > lm[L_SHOULDER].x

    # ── 생체역학 손목 각도 및 라켓 방향 계산 ──
    wx, wy = r_wrist if is_right_handed else l_wrist
    ex, ey = r_elbow if is_right_handed else l_elbow
    dx, dy = wx - ex, wy - ey
    arm_len = math.sqrt(dx * dx + dy * dy) + 1e-6
    nx, ny = dx / arm_len, dy / arm_len
    forearm_angle = math.atan2(ny, nx)

    # 손목 래그(Wrist Lag) 및 와이퍼 프로네이션 각도 오프셋 정의
    global _racket_offset_prev
    if is_serve:
        target_offset = 0.0
    else:
        if -45 < k < 40:
            if k < -15:
                # 준비 자세: 오프셋 없음
                target_offset = 0.0
            elif k < -10:
                # 레이백 진입
                frac = (k + 15) / 5.0
                target_offset = -1.2 * frac
            elif k < -2:
                # 테이크백 및 드롭 (최대 레이백)
                frac = (k + 10) / 8.0
                target_offset = -1.2 * (1.0 - frac) + -2.8 * frac
            elif k <= 0:
                # 임팩트 전 스냅 스윙 (가속 단계)
                frac = (k + 2) / 2.0
                target_offset = -2.8 * (1.0 - frac) + -0.6 * frac
            elif k <= 6:
                # 임팩트 후 와이퍼 프로네이션 감아올리기
                frac = (k / 6.0)
                target_offset = -0.6 * (1.0 - frac) + 1.0 * frac
            elif k < 30:
                # 천천히 감쇠
                frac = (k - 6) / 24.0
                target_offset = 1.0 * (1.0 - frac)
            else:
                target_offset = 0.0
        else:
            target_offset = 0.0

    # Multiply target_offset by lag_scale
    target_offset_scaled = target_offset * lag_scale

    if _racket_offset_prev is None:
        _racket_offset_prev = target_offset_scaled
    else:
        # 가속 및 임팩트 스냅 구간(-2~8프레임)에는 스냅 응답성을 최대화하기 위해 필터 지연 최소화
        alpha_offset = 0.75 if -2 <= k <= 8 else 0.25
        _racket_offset_prev = _racket_offset_prev + alpha_offset * (target_offset_scaled - _racket_offset_prev)

    pronation_angle = forearm_angle + (_racket_offset_prev if is_right_handed else -_racket_offset_prev)
    nx_racket = math.cos(pronation_angle)
    ny_racket = math.sin(pronation_angle)

    # ── 3D 라켓 헤드 모핑 제어 (가로세로 비율 조절) ──
    global _racket_face_prev
    if is_serve:
        target_factor = 1.0
    else:
        if -45 < k < 40:
            if k < -6:
                target_factor = 1.0
            elif k < 0:
                # 임팩트 직전 닫힘 (Drop 엣지온)
                frac = (k + 6) / 6.0
                target_factor = 1.0 * (1.0 - frac) + (0.20 / 1.15) * frac
            elif k < 6:
                # 임팩트 후 프로네이션 회전 (엣지온으로 전개)
                target_factor = (0.20 + 0.35 * (k / 6.0)) / 1.15
            elif k < 30:
                target_factor = 0.55 / 1.15
            else:
                # 다시 중립 준비 상태로 완만하게 회복
                frac = (k - 30) / 10.0
                target_factor = (0.55 / 1.15) * (1.0 - frac) + 1.0 * frac
        else:
            target_factor = 1.0

    if _racket_face_prev is None:
        _racket_face_prev = target_factor
    else:
        alpha_face = 0.12
        _racket_face_prev = _racket_face_prev + alpha_face * (target_factor - _racket_face_prev)

    # ── 다중 잔상(Stroboscopic Ghosting) 렌더링 ──
    current_state = {
        "l_shoulder": l_shoulder, "r_shoulder": r_shoulder,
        "l_elbow": l_elbow, "r_elbow": r_elbow,
        "l_wrist": l_wrist, "r_wrist": r_wrist,
        "l_hip": l_hip, "r_hip": r_hip,
        "l_knee": l_knee, "r_knee": r_knee,
        "l_ankle": l_ankle, "r_ankle": r_ankle,
        "l_heel": l_heel, "r_heel": r_heel,
        "l_foot_idx": l_foot_idx, "r_foot_idx": r_foot_idx,
        "nx_racket": nx_racket, "ny_racket": ny_racket,
        "racket_face_ratio": _racket_face_prev,
        "head_r": head_r, "hand_r": hand_r,
        "is_back_view": is_back_view
    }

    if strobe_history is not None:
        # 임팩트 이후(k > 0)에는 이전 잔상이 나타나지 않도록 히스토리를 초기화
        if 0 < k < 9999:
            strobe_history.clear()
        n_hist = len(strobe_history)
        if n_hist >= strobe_step:
            alpha_base = 0.25  # 가장 최신 잔상의 투명도
            for idx in range(0, n_hist, strobe_step):
                # 오래될수록 흐려지는 그라데이션
                alpha = 0.05 + (alpha_base - 0.05) * (idx / max(n_hist - 1, 1))
                overlay = canvas.copy()
                draw_ghost_figure(overlay, strobe_history[idx], is_right_handed)
                cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

        strobe_history.append(current_state)
        if len(strobe_history) > strobe_frames:
            strobe_history.pop(0)

    # ── 네온 스윙 궤적 계산 및 그리기 ──
    racket_wrist = r_wrist if is_right_handed else l_wrist
    grip_length = int(head_r * 1.2)
    frame_ry = int(head_r * 1.5)
    grip_end_x = int(racket_wrist[0] + nx_racket * grip_length)
    grip_end_y = int(racket_wrist[1] + ny_racket * grip_length)
    head_cx_racket = int(grip_end_x + nx_racket * frame_ry)
    head_cy_racket = int(grip_end_y + ny_racket * frame_ry)
    racket_center = (head_cx_racket, head_cy_racket)

    if racket_trail is not None:
        racket_trail.append(racket_center)
        if len(racket_trail) > 20:
            racket_trail.pop(0)
        draw_glowing_trail(canvas, racket_trail, color=(0, 220, 255))  # 네온 옐로우 (BGR)
        
    if hand_trail is not None:
        hand_trail.append(racket_wrist)
        if len(hand_trail) > 20:
            hand_trail.pop(0)
        draw_glowing_trail(canvas, hand_trail, color=(255, 150, 50))   # 네온 시안 (BGR)

    # ── 1. 맨 밑바탕: 그림자 그리기 ──
    draw_shadow(canvas, l_ankle, r_ankle)

    # ── 2. 다리 그리기 ──
    draw_body_line(canvas, l_hip, l_knee)
    draw_body_line(canvas, l_knee, l_ankle)
    draw_body_line(canvas, r_hip, r_knee)
    draw_body_line(canvas, r_knee, r_ankle)
    joint_r = max(LIMB_THICKNESS // 2, int(2 * LW))
    cv2.circle(canvas, l_knee, joint_r, BODY_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, r_knee, joint_r, BODY_COLOR, -1, cv2.LINE_AA)

    # ── 3. Z-depth 기준 레이어 렌더링 ──
    # 양손 그립 모드: 왼손을 라켓 그립 상단(오른손 위)에 고정
    grip_length_px = int(head_r * 1.2)
    if two_handed and is_right_handed:
        l_wrist_render = (
            int(wx + nx_racket * grip_length_px * 0.65),
            int(wy + ny_racket * grip_length_px * 0.65)
        )
    else:
        l_wrist_render = l_wrist

    def draw_l_upper_arm():
        draw_body_line(canvas, l_shoulder, l_elbow)
        cv2.circle(canvas, l_elbow, joint_r, BODY_COLOR, -1, cv2.LINE_AA)

    def draw_l_forearm():
        draw_body_line(canvas, l_elbow, l_wrist_render)
        draw_hand(canvas, l_wrist_render, hand_r)
        if not is_right_handed:
            draw_racket(canvas, l_wrist_render, l_elbow, head_r, nx_racket, ny_racket, _racket_face_prev)

    def draw_r_upper_arm():
        draw_body_line(canvas, r_shoulder, r_elbow)
        cv2.circle(canvas, r_elbow, joint_r, BODY_COLOR, -1, cv2.LINE_AA)

    def draw_r_forearm():
        draw_body_line(canvas, r_elbow, r_wrist)
        draw_hand(canvas, r_wrist, hand_r)
        if is_right_handed:
            draw_racket(canvas, r_wrist, r_elbow, head_r, nx_racket, ny_racket, _racket_face_prev)

    def draw_trunk():
        draw_shorts(canvas, l_hip, r_hip, l_knee, r_knee, head_r)
        draw_pentagon_torso(canvas, neck, l_shoulder, r_shoulder, l_hip, r_hip)
        draw_body_line(canvas, neck, tuple(head_bottom), thickness=NECK_THICKNESS)
        draw_head(canvas, head_cx, head_cy, head_r)

    l_upper_z = max(lm[L_SHOULDER].z, lm[L_ELBOW].z)
    l_forearm_z = lm[L_ELBOW].z
    r_upper_z = max(lm[R_SHOULDER].z, lm[R_ELBOW].z)
    r_forearm_z = lm[R_ELBOW].z
    
    if is_back_view:
        trunk_z = min(l_upper_z, l_forearm_z, r_upper_z, r_forearm_z) - 0.1
    else:
        trunk_z = max(l_upper_z, l_forearm_z, r_upper_z, r_forearm_z) + 0.1

    # 동작 단계(Phase) 감지를 미리 수행하여 Z-depth 보정에 사용
    elbow_joint = r_elbow if is_right_handed else l_elbow
    elbow_p1 = r_shoulder if is_right_handed else l_shoulder
    elbow_p2 = r_wrist if is_right_handed else l_wrist
    
    h_elbow_angle = calculate_angle_2d(elbow_p1, elbow_joint, elbow_p2)
    if is_serve:
        raw_phase = detect_serve_phase(lm, is_right_handed, h_elbow_angle)
    else:
        raw_phase = detect_groundstroke_phase(lm, is_right_handed, h_elbow_angle)
        
    if phase_smoother is not None:
        current_phase = phase_smoother.add_and_get(raw_phase)
    else:
        current_phase = raw_phase

    # 타격 팔이 몸/머리 뒤로 넘어갔는지 검사
    # 방법 1: 기하학적 조건 (서브 팔로우스루 등, 팔이 목 반대편으로 넘어갈 때)
    is_arm_behind_geom = False
    if is_right_handed:
        if r_wrist[0] < neck[0] and r_elbow[1] < r_shoulder[1] + int(30 * LW):
            is_arm_behind_geom = True
    else:
        if l_wrist[0] > neck[0] and l_elbow[1] < l_shoulder[1] + int(30 * LW):
            is_arm_behind_geom = True

    # 방법 2: MediaPipe z-depth 기반 감지 (백핸드 팔로우스루 후 몸 뒤로 넘어가는 경우)
    # z가 양수일수록 카메라에서 멀리 있음. 팔 평균 z가 상체 z보다 유의미하게 크면 뒤에 있는 것
    torso_z_mid = (lm[L_SHOULDER].z + lm[R_SHOULDER].z) / 2.0
    if is_right_handed:
        arm_z_avg = (lm[R_WRIST].z + lm[R_ELBOW].z) / 2.0
    else:
        arm_z_avg = (lm[L_WRIST].z + lm[L_ELBOW].z) / 2.0
    is_arm_behind_z = arm_z_avg > torso_z_mid + 0.10

    is_arm_behind = is_arm_behind_geom or is_arm_behind_z

    h_wrist_y = r_wrist[1] if is_right_handed else l_wrist[1]
    is_occluded = False
    if (is_serve and current_phase == "Racket Drop") or (is_arm_behind_geom and is_arm_behind_z):
        if h_wrist_y > head_cy:
            is_occluded = True
    elif is_arm_behind_z:
        # z-depth 감지의 경우: 손목 높이와 무관하게 오클루전 적용 (백핸드 팔로우스루 포함)
        is_occluded = True

    if is_occluded:
        # 양손 백핸드처럼 양팔이 함께 뒤로 넘어가는 경우를 위해 두 팔 전체 최솟값 기준으로 몸통 우선 렌더링
        trunk_z = min(l_upper_z, l_forearm_z, r_upper_z, r_forearm_z) - 0.1

    draw_tasks = [
        (l_upper_z, draw_l_upper_arm),
        (l_forearm_z, draw_l_forearm),
        (r_upper_z, draw_r_upper_arm),
        (r_forearm_z, draw_r_forearm),
        (trunk_z, draw_trunk)
    ]
    draw_tasks.sort(key=lambda x: x[0], reverse=True)
    for depth, draw_func in draw_tasks:
        draw_func()

    # ── 4. 신발 그리기 ──
    draw_shoe(canvas, l_ankle, l_heel, l_foot_idx, l_knee, head_r, is_back_view, side_key="left")
    draw_shoe(canvas, r_ankle, r_heel, r_foot_idx, r_knee, head_r, is_back_view, side_key="right")

    # ── 5. 관절 각도 계산 및 오버레이 그리기 ──
    draw_angle_tasks = []
    
    elbow_z = lm[R_ELBOW].z if is_right_handed else lm[L_ELBOW].z
    
    if elbow_z <= trunk_z:
        task1 = draw_angle_overlay(canvas, elbow_joint, elbow_p1, elbow_p2, color=(0, 165, 255))
        draw_angle_tasks.append(task1)
    
    task2 = draw_angle_overlay(canvas, l_knee, l_hip, l_ankle, color=(50, 220, 50))
    task3 = draw_angle_overlay(canvas, r_knee, r_hip, r_ankle, color=(50, 220, 50))
    draw_angle_tasks.extend([task2, task3])

    draw_texts_pil(canvas, draw_angle_tasks, font_size=max(int(10 * LW), 10))

    return current_phase


def correct_leg_swaps(landmarks, prev_landmarks):
    if prev_landmarks is None:
        return landmarks
    pairs = [
        (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
    ]
    dist_no_swap = 0.0
    for l_idx, r_idx in pairs:
        pl = prev_landmarks[l_idx]
        pr = prev_landmarks[r_idx]
        cl = landmarks[l_idx]
        cr = landmarks[r_idx]
        dist_no_swap += (cl.x - pl.x)**2 + (cl.y - pl.y)**2
        dist_no_swap += (cr.x - pr.x)**2 + (cr.y - pr.y)**2

    dist_swap = 0.0
    for l_idx, r_idx in pairs:
        pl = prev_landmarks[l_idx]
        pr = prev_landmarks[r_idx]
        cl = landmarks[l_idx]
        cr = landmarks[r_idx]
        dist_swap += (cl.x - pr.x)**2 + (cl.y - pr.y)**2
        dist_swap += (cr.x - pl.x)**2 + (cr.y - pl.y)**2

    if dist_swap < dist_no_swap and (dist_no_swap - dist_swap) > 0.015:
        for l_idx, r_idx in pairs:
            lx, ly, lz = landmarks[l_idx].x, landmarks[l_idx].y, landmarks[l_idx].z
            rx, ry, rz = landmarks[r_idx].x, landmarks[r_idx].y, landmarks[r_idx].z
            landmarks[l_idx].x, landmarks[l_idx].y, landmarks[l_idx].z = rx, ry, rz
            landmarks[r_idx].x, landmarks[r_idx].y, landmarks[r_idx].z = lx, ly, lz

    return landmarks


def correct_leg_orientation_global(all_smoothed_landmarks):
    """발목 방향 다수결로 다리 랜드마크 좌우를 교정.

    발목 L<R 비율로 기준 방향 결정 후, 기준과 다르고 차이가 임계값 이상인
    프레임만 다리 쌍(23-32) 좌우 교환.
    THRESHOLD: 두 발목이 충분히 벌어진 경우에만 교정해 점프 중 미세 교차 오탐 방지.
    경계 전후 스무딩은 이후 ±LEG_WIN 이동평균 안정화 단계에서 처리.
    """
    L_ANKLE, R_ANKLE = 27, 28
    LEG_PAIRS = [(23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
    THRESHOLD = 0.03  # 발목 차이 최소 임계값 (이하는 교정 안 함)

    valid = [lms for lms in all_smoothed_landmarks if lms is not None]
    back_count = sum(1 for lms in valid if lms[L_ANKLE].x < lms[R_ANKLE].x)
    expected_back = back_count / len(valid) > 0.5

    corrected = 0
    for lms in all_smoothed_landmarks:
        if lms is None:
            continue
        diff = lms[L_ANKLE].x - lms[R_ANKLE].x
        ankle_back = diff < 0
        if ankle_back != expected_back and abs(diff) >= THRESHOLD:
            for l_idx, r_idx in LEG_PAIRS:
                l, r = lms[l_idx], lms[r_idx]
                lms[l_idx] = SimpleNamespace(x=r.x, y=r.y, z=r.z, v=getattr(r, 'v', 1.0))
                lms[r_idx] = SimpleNamespace(x=l.x, y=l.y, z=l.z, v=getattr(l, 'v', 1.0))
            corrected += 1

    direction = "등향(L<R)" if expected_back else "정면향(L>R)"
    print(f"[다리 방향 교정] 기준={direction}, {corrected}프레임 교정 (임계값 {THRESHOLD})")
    return all_smoothed_landmarks


# ─────────────────────────────────────────
# TrackNet 딥러닝 공 추적 (v10 신규)
# ─────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _ConvBlock(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        import torch.nn as nn
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch)
        )
    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module if _TORCH_AVAILABLE else object):
    """TrackNet — 3프레임 9ch 입력 → 공 위치 히트맵"""
    def __init__(self, out_channels=256):
        super().__init__()
        import torch.nn as nn
        self.out_channels = out_channels
        self.conv1  = _ConvBlock(9, 64);   self.conv2  = _ConvBlock(64, 64)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.conv3  = _ConvBlock(64, 128); self.conv4  = _ConvBlock(128, 128)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.conv5  = _ConvBlock(128, 256); self.conv6 = _ConvBlock(256, 256); self.conv7 = _ConvBlock(256, 256)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.conv8  = _ConvBlock(256, 512); self.conv9 = _ConvBlock(512, 512); self.conv10 = _ConvBlock(512, 512)
        self.ups1   = nn.Upsample(scale_factor=2)
        self.conv11 = _ConvBlock(512, 256); self.conv12 = _ConvBlock(256, 256); self.conv13 = _ConvBlock(256, 256)
        self.ups2   = nn.Upsample(scale_factor=2)
        self.conv14 = _ConvBlock(256, 128); self.conv15 = _ConvBlock(128, 128)
        self.ups3   = nn.Upsample(scale_factor=2)
        self.conv16 = _ConvBlock(128, 64);  self.conv17 = _ConvBlock(64, 64)
        self.conv18 = _ConvBlock(64, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, testing=False):
        b = x.size(0)
        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = self.pool3(self.conv7(self.conv6(self.conv5(x))))
        x = self.conv10(self.conv9(self.conv8(x)))
        x = self.conv13(self.conv12(self.conv11(self.ups1(x))))
        x = self.conv15(self.conv14(self.ups2(x)))
        x = self.conv18(self.conv17(self.conv16(self.ups3(x))))
        out = x.reshape(b, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out


def _tracknet_postprocess(feature_map, w=640):
    """히트맵 argmax 배열 → (x, y) 0~1 정규화 좌표"""
    h = feature_map.shape[0] // w
    fm = (feature_map * 255).reshape(h, w).astype(np.uint8)
    _, hm = cv2.threshold(fm, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(hm, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                               param1=50, param2=2, minRadius=2, maxRadius=7)
    if circles is not None and len(circles) == 1:
        return circles[0][0][0] / w, circles[0][0][1] / h  # 정규화 좌표
    return None, None


def detect_impacts_from_tracknet(video_path, fps, n_frames, model_path, min_gap_sec=1.5):
    """TrackNet으로 공 추적 → x방향 속도 반전 = 임팩트"""
    if not _TORCH_AVAILABLE:
        print("[!] PyTorch 미설치 — pip install torch 실행 후 재시도")
        return []

    import torch
    print(f"[i] TrackNet 공 추적 분석 중... (model: {model_path})")

    device = 'cpu'
    model = BallTrackerNet()
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"[!] 모델 로드 실패: {e}")
        return []
    model.to(device).eval()

    cap = cv2.VideoCapture(video_path)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    TW, TH = 320, 180  # 절반 해상도로 4배 빠름 (8의 배수)
    ball_track = [(None, None)] * 2  # 첫 2프레임은 이전 프레임이 없어 건너뜀

    with torch.no_grad():
        for i in range(2, len(frames)):
            imgs = np.concatenate([
                cv2.resize(frames[i],   (TW, TH)),
                cv2.resize(frames[i-1], (TW, TH)),
                cv2.resize(frames[i-2], (TW, TH)),
            ], axis=2).astype(np.float32) / 255.0
            inp = torch.from_numpy(np.rollaxis(imgs, 2, 0)[None]).float()
            out = model(inp, testing=True)
            feat = out.argmax(dim=1).detach().cpu().numpy()[0]
            xn, yn = _tracknet_postprocess(feat, w=TW)

            if xn is not None:
                # 정규화 좌표 → 실제 영상 픽셀 좌표
                ball_track.append((xn * actual_w, yn * actual_h))
            else:
                ball_track.append((None, None))

            if i % 100 == 0:
                det = sum(1 for p in ball_track if p[0] is not None)
                print(f"  TrackNet: frame {i}/{n_frames} (공 감지: {det}프레임)")

    det = sum(1 for p in ball_track if p[0] is not None)
    print(f"[i] 공 감지 프레임: {det}/{n_frames} ({det/n_frames*100:.1f}%)")

    if det < n_frames * 0.03:
        print("[!] 공 감지율 3% 미만")
        return []

    # x방향 속도 (window=3)
    win = 3
    vx = [None] * len(ball_track)
    for i in range(win, len(ball_track) - win):
        pa, pb = ball_track[i - win], ball_track[i + win]
        if pa[0] and pb[0]:
            vx[i] = pb[0] - pa[0]

    # 속도 반전 = 임팩트
    min_gap = int(min_gap_sec * fps)
    impact_points = []
    for i in range(win, len(vx) - 1):
        if vx[i] is None or vx[i + 1] is None:
            continue
        if vx[i] * vx[i + 1] < 0 and (abs(vx[i]) + abs(vx[i + 1])) > 15:
            if not impact_points or i - impact_points[-1] >= min_gap:
                impact_points.append(i)
                print(f"[✓] TrackNet impact: f{i} "
                      f"(t={i/fps:.2f}s, vx: {vx[i]:+.1f}→{vx[i+1]:+.1f}, "
                      f"pos={ball_track[i]})")

    return impact_points


# ─────────────────────────────────────────
# 공 추적 기반 임팩트 감지 (HSV 색상 필터)
# ─────────────────────────────────────────

def detect_impacts_from_ball(video_path, fps, n_frames, min_gap_sec=1.5):
    """HSV로 테니스 공 추적 → x방향 속도 반전 = 임팩트"""
    print("[i] 공 추적 분석 중...")

    lower_ball = np.array([20, 60, 60])
    upper_ball = np.array([50, 255, 255])

    cap = cv2.VideoCapture(video_path)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 상단 20% 영역은 자막/로고 오탐 방지를 위해 마스크에서 제외
    roi_top = int(frame_h * 0.20)

    ball_positions = []  # (cx, cy) 또는 None
    kernel = np.ones((3, 3), np.uint8)
    prev_pos = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_ball, upper_ball)
        # 상단 ROI 제외
        mask[:roi_top, :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_pos, best_circ = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 8 or area > 1500:
                continue
            peri = cv2.arcLength(c, True)
            if peri < 1:
                continue
            circ = 4 * math.pi * area / (peri * peri)
            if circ <= 0.45:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            # 이전 공 위치에서 120픽셀 이내인 후보만 유효 (연속성)
            if prev_pos is not None:
                dist = math.sqrt((cx - prev_pos[0])**2 + (cy - prev_pos[1])**2)
                if dist > 120:
                    continue
            if circ > best_circ:
                best_circ = circ
                best_pos = (cx, cy)

        # 이전 위치 없을 때는 연속성 무시하고 원형도만으로 선택 (첫 감지용)
        if best_pos is None and prev_pos is None:
            best_circ2 = 0.0
            for c in contours:
                area = cv2.contourArea(c)
                if area < 8 or area > 1500:
                    continue
                peri = cv2.arcLength(c, True)
                if peri < 1:
                    continue
                circ = 4 * math.pi * area / (peri * peri)
                if circ > best_circ2:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        best_circ2 = circ
                        best_pos = (M["m10"] / M["m00"], M["m01"] / M["m00"])
            if best_circ2 < 0.45:
                best_pos = None

        ball_positions.append(best_pos)
        prev_pos = best_pos if best_pos else prev_pos  # 미감지 시 이전 위치 유지 (연속성용)
        frame_idx += 1
        if frame_idx % 100 == 0:
            detected = sum(1 for p in ball_positions if p is not None)
            print(f"  Ball tracking: frame {frame_idx}/{n_frames} (감지: {detected}프레임)")

    cap.release()

    detected_count = sum(1 for p in ball_positions if p is not None)
    print(f"[i] 공 감지 프레임: {detected_count}/{n_frames} ({detected_count/n_frames*100:.1f}%)")

    if detected_count < n_frames * 0.03:
        print("[!] 공 감지율 3% 미만 — HSV 범위가 맞지 않을 수 있음")
        return []

    # x방향 속도 계산 (window=2프레임)
    win = 2
    vx = [None] * len(ball_positions)
    for i in range(win, len(ball_positions) - win):
        pa = ball_positions[i - win]
        pb = ball_positions[i + win]
        if pa and pb:
            vx[i] = pb[0] - pa[0]

    # x방향 속도 반전 = 임팩트 (임계: 총 변화량 30픽셀 이상)
    min_gap = int(min_gap_sec * fps)
    impact_points = []
    for i in range(win, len(vx) - 1):
        if vx[i] is None or vx[i + 1] is None:
            continue
        if vx[i] * vx[i + 1] < 0 and (abs(vx[i]) + abs(vx[i + 1])) > 30:
            if not impact_points or i - impact_points[-1] >= min_gap:
                impact_points.append(i)
                print(f"[✓] Ball impact at Frame {i} "
                      f"(t={i/fps:.2f}s, vx: {vx[i]:+.1f}→{vx[i+1]:+.1f}, "
                      f"pos={ball_positions[i]})")

    return impact_points


# ─────────────────────────────────────────
# 오디오 타구음 기반 임팩트 감지
# ─────────────────────────────────────────

def detect_impacts_from_audio(video_path, fps, n_frames, min_gap_sec=1.0, threshold_ratio=0.20):
    import tempfile, wave
    print("[i] 오디오 타구음 분석 중...")
    tmp_wav = tempfile.mktemp(suffix='.wav')
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-ac', '1', '-ar', '44100',
            '-c:a', 'pcm_s16le', tmp_wav
        ], capture_output=True, check=True)
    except Exception as e:
        print(f"[!] 오디오 추출 실패: {e}")
        return []

    with wave.open(tmp_wav, 'rb') as wf:
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    os.remove(tmp_wav)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # 단시간 에너지 계산 (20ms 윈도우, 5ms 홉)
    win = int(sample_rate * 0.020)
    hop = int(sample_rate * 0.005)
    n_wins = (len(samples) - win) // hop
    energy = np.array([
        np.sum(samples[i*hop:i*hop+win] ** 2) / win
        for i in range(n_wins)
    ])

    # 에너지 상승률 (onset 강도)
    onset = np.diff(energy)
    onset = np.clip(onset, 0, None)
    if onset.max() > 0:
        onset /= onset.max()

    # 피크 탐색 (최소 간격: min_gap_sec)
    min_gap_wins = int(min_gap_sec * sample_rate / hop)
    peaks = []
    i = 0
    while i < len(onset):
        if onset[i] >= threshold_ratio:
            j = i
            while j < len(onset) and onset[j] >= threshold_ratio * 0.4:
                j += 1
            pk = i + int(np.argmax(onset[i:j]))
            if not peaks or pk - peaks[-1] > min_gap_wins:
                peaks.append(pk)
            i = j
        else:
            i += 1

    # 오디오 피크 → 비디오 프레임 + onset 값 변환
    video_frames = []
    onset_values = []
    for pk in peaks:
        t = pk * hop / sample_rate
        f = min(int(t * fps), n_frames - 1)
        video_frames.append(f)
        onset_values.append(float(onset[pk]))
        print(f"[✓] Audio impact at Frame {f} (t={t:.3f}s, onset={onset[pk]:.3f})")

    return video_frames, onset_values


# ─────────────────────────────────────────
# 팔신장 임팩트 감지 (독립 함수 — 앙상블용)
# ─────────────────────────────────────────

def detect_impacts_arm_extension(all_smoothed_landmarks, is_right_handed):
    """어깨-손목 신장도 기반 임팩트 감지. (extensions 리스트, impact_points 리스트) 반환."""
    h_shoulder_idx = R_SHOULDER if is_right_handed else L_SHOULDER
    h_wrist_idx    = R_WRIST    if is_right_handed else L_WRIST
    n = len(all_smoothed_landmarks)
    win = 3

    extensions = []
    for fi in range(n):
        sm = all_smoothed_landmarks[fi]
        if sm is None:
            extensions.append(0.0)
            continue
        s = sm[h_shoulder_idx]
        w = sm[h_wrist_idx]
        extensions.append(math.sqrt((w.x - s.x)**2 + (w.y - s.y)**2))

    abs_velocities = []
    for fi in range(n):
        a, b = fi - win, fi + win
        if a < 0 or b >= n or all_smoothed_landmarks[a] is None or all_smoothed_landmarks[b] is None:
            abs_velocities.append(0.0)
            continue
        wa = all_smoothed_landmarks[a][h_wrist_idx]
        wb = all_smoothed_landmarks[b][h_wrist_idx]
        abs_velocities.append(math.sqrt((wb.x - wa.x)**2 + (wb.y - wa.y)**2) / (2 * win))

    max_abs = max(abs_velocities) if abs_velocities else 0
    vel_threshold = max_abs * 0.12

    in_swing, swing_start = False, 0
    swing_windows = []
    for fi in range(n):
        if not in_swing and abs_velocities[fi] >= vel_threshold:
            in_swing, swing_start = True, fi
        elif in_swing and abs_velocities[fi] < vel_threshold * 0.25:
            if fi - swing_start >= 6:
                swing_windows.append((swing_start, fi))
            in_swing = False
    if in_swing:
        swing_windows.append((swing_start, n - 1))

    merged = []
    for ws, we in swing_windows:
        if merged and ws - merged[-1][1] <= 20:
            merged[-1] = (merged[-1][0], we)
        else:
            merged.append((ws, we))

    impact_points = []
    for ws, we in merged:
        ws_ext = max(0, ws - 15)
        window_exts = extensions[ws_ext:we + 1]
        max_ext = max(window_exts) if window_exts else 0
        threshold_ext = max_ext * 0.50

        impact_f = we
        for fi in range(ws_ext, we + 1):
            if extensions[fi] >= threshold_ext:
                impact_f = fi
                break

        sm = all_smoothed_landmarks[impact_f]
        if sm is not None:
            wrist_y    = sm[h_wrist_idx].y
            shoulder_y = sm[h_shoulder_idx].y
            if wrist_y < shoulder_y - 0.04:
                print(f"[i] Skipped frame {impact_f}: wrist above shoulder")
                continue

        impact_points.append(impact_f)
        print(f"[✓] Arm-extension impact at Frame {impact_f} "
              f"(ext={extensions[impact_f]:.4f}, window={ws_ext}-{we})")

    min_gap = 150
    deduped = []
    for imp in sorted(impact_points):
        if deduped and imp - deduped[-1] < min_gap:
            if extensions[imp] > extensions[deduped[-1]]:
                print(f"[i] Replaced f{deduped[-1]} with f{imp} (higher extension)")
                deduped[-1] = imp
            else:
                print(f"[i] Removed f{imp}: too close to f{deduped[-1]} (lower extension)")
        else:
            deduped.append(imp)

    return deduped, extensions


def detect_impacts_audio_validated(video_path, fps, n_frames, all_smoothed_landmarks, is_right_handed):
    """오디오 타구음 + 랜드마크 이중검증.
    - 오디오 피크를 먼저 클러스터링 (4초 이내 = 같은 샷 묶음)
    - 클러스터당 최고 점수 프레임 1개 선택
    - 검색 창: [-20, +5] 비대칭
    - 점수: xoff + 손목속도 가중치 + 팔꿈치 증가 조건
    """
    raw_frames, onsets = detect_impacts_from_audio(video_path, fps, n_frames)
    if not raw_frames or not all_smoothed_landmarks:
        return []

    # ── 상대 임계값 필터: max_onset * 0.25 이상만 사용 ──
    # 약한 에코/잡음 제거, 강한 타구음만 남김
    max_onset = max(onsets) if onsets else 0
    rel_threshold = max(0.20, max_onset * 0.25)
    audio_candidates = [f for f, o in zip(raw_frames, onsets) if o >= rel_threshold]
    print(f"[i] 오디오 필터: max_onset={max_onset:.3f}, threshold={rel_threshold:.3f} → {len(audio_candidates)}/{len(raw_frames)}개 통과: {audio_candidates}")

    if not audio_candidates:
        return []

    # ── 오디오 피크 클러스터링: fps*4(4초) 이내는 같은 샷으로 묶음 ──
    CLUSTER_GAP = int(fps * 4)
    clusters = []
    for af in sorted(audio_candidates):
        if clusters and af - clusters[-1][-1] < CLUSTER_GAP:
            clusters[-1].append(af)
        else:
            clusters.append([af])
    print(f"[i] 오디오 클러스터 {len(clusters)}개: {clusters}")

    h_s = R_SHOULDER if is_right_handed else L_SHOULDER
    h_w = R_WRIST    if is_right_handed else L_WRIST
    h_e = R_ELBOW    if is_right_handed else L_ELBOW
    VEL_WIN  = 3
    LOOK_BACK = 5

    def best_frame_for_audio(af):
        """오디오 피크 af 주변에서 최적 임팩트 프레임과 점수 반환."""
        best_f, best_s = af, -1.0
        for fi in range(max(0, af - 20), min(n_frames, af + 3)):
            sm = all_smoothed_landmarks[fi]
            if sm is None:
                continue
            s, w, e = sm[h_s], sm[h_w], sm[h_e]
            if abs(w.x - s.x) <= 0.03:
                continue
            if not (s.y - 0.05 <= w.y <= s.y + 0.30):
                continue
            ea = calculate_angle_2d((s.x, s.y), (e.x, e.y), (w.x, w.y))
            if ea < 100.0:
                continue
            # 팔꿈치 증가 조건: 이전 5프레임보다 현재 각도가 높아야 (팔로우스루 제거)
            prev_angles = [
                calculate_angle_2d(
                    (all_smoothed_landmarks[p][h_s].x, all_smoothed_landmarks[p][h_s].y),
                    (all_smoothed_landmarks[p][h_e].x, all_smoothed_landmarks[p][h_e].y),
                    (all_smoothed_landmarks[p][h_w].x, all_smoothed_landmarks[p][h_w].y),
                )
                for p in range(max(0, fi - 5), fi)
                if all_smoothed_landmarks[p] is not None
            ]
            if prev_angles and ea <= max(prev_angles):
                continue
            x_off = abs(w.x - s.x)
            a, b = fi - VEL_WIN, fi + VEL_WIN
            if 0 <= a and b < n_frames and all_smoothed_landmarks[a] and all_smoothed_landmarks[b]:
                wa, wb = all_smoothed_landmarks[a][h_w], all_smoothed_landmarks[b][h_w]
                vel = math.sqrt((wb.x - wa.x)**2 + (wb.y - wa.y)**2) / (2 * VEL_WIN)
            else:
                vel = 0.0
            # elbow(153°→높을수록) + xoff + vel 복합 점수
            score = ea * 0.001 + x_off + vel * 7.5
            if score > best_s:
                best_s, best_f = score, fi
        return best_f, best_s

    # ── 클러스터당 최고 점수 프레임 1개 선택 ──
    result = []
    for cluster in clusters:
        cluster_best_f, cluster_best_s = None, -1.0
        for af in cluster:
            bf, bs = best_frame_for_audio(af)
            if bs > cluster_best_s:
                cluster_best_s, cluster_best_f = bs, bf
        if cluster_best_s > 0:
            result.append(cluster_best_f)
            print(f"[✓] 오디오+랜드마크 검증: f{cluster_best_f} "
                  f"(cluster={cluster}, score={cluster_best_s:.3f})")
        else:
            print(f"[i] 클러스터 {cluster}: 랜드마크 검증 실패")

    return result


def _elbow_angle_at(lms, fi, is_right_handed):
    """fi 프레임의 팔꿈치 각도(도) 반환. 랜드마크 없으면 0."""
    sm = lms[fi]
    if sm is None:
        return 0.0
    h_s = R_SHOULDER if is_right_handed else L_SHOULDER
    h_e = R_ELBOW    if is_right_handed else L_ELBOW
    h_w = R_WRIST    if is_right_handed else L_WRIST
    s, e, w = sm[h_s], sm[h_e], sm[h_w]
    v1 = (s.x - e.x, s.y - e.y)
    v2 = (w.x - e.x, w.y - e.y)
    denom = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2) + 1e-6
    cos_t = (v1[0]*v2[0] + v1[1]*v2[1]) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))


def _wrist_speed_at(lms, fi, is_right_handed, win=3):
    """fi 프레임의 손목 속도 반환 (±win 프레임 차분)."""
    n = len(lms)
    a, b = fi - win, fi + win
    if a < 0 or b >= n or lms[a] is None or lms[b] is None:
        return 0.0
    h_w = R_WRIST if is_right_handed else L_WRIST
    wa, wb = lms[a][h_w], lms[b][h_w]
    return math.sqrt((wb.x - wa.x)**2 + (wb.y - wa.y)**2) / (2 * win)


def detect_impacts_ensemble(video_path, fps, n_frames, all_smoothed_landmarks, is_right_handed):
    """통합 앙상블: 오디오+랜드마크 이중검증(1순위) → 팔꿈치조건 팔신장(폴백)."""
    print("[i] 통합 앙상블 감지...")

    # 1순위: 오디오 + 랜드마크 이중검증 (v12.1 방식)
    result = detect_impacts_audio_validated(
        video_path, fps, n_frames, all_smoothed_landmarks, is_right_handed
    )
    if result:
        print(f"[✓] 오디오+랜드마크 이중검증 성공 → {result}")
        return result

    # 2순위: 팔꿈치 조건(≥140°+감소중) + 복합점수 팔신장 (v12.3 방식)
    print("[i] 오디오+랜드마크 실패 → 팔꿈치 조건 팔신장 폴백...")
    h_s = R_SHOULDER if is_right_handed else L_SHOULDER
    h_w = R_WRIST    if is_right_handed else L_WRIST
    n = len(all_smoothed_landmarks)
    win = 3

    extensions, abs_vels = [], []
    for fi in range(n):
        sm = all_smoothed_landmarks[fi]
        if sm is None:
            extensions.append(0.0); abs_vels.append(0.0); continue
        s, w = sm[h_s], sm[h_w]
        extensions.append(math.sqrt((w.x - s.x)**2 + (w.y - s.y)**2))
        a, b = fi - win, fi + win
        if a < 0 or b >= n or all_smoothed_landmarks[a] is None or all_smoothed_landmarks[b] is None:
            abs_vels.append(0.0)
        else:
            wa, wb = all_smoothed_landmarks[a][h_w], all_smoothed_landmarks[b][h_w]
            abs_vels.append(math.sqrt((wb.x - wa.x)**2 + (wb.y - wa.y)**2) / (2 * win))

    max_v = max(abs_vels) if abs_vels else 0
    vel_thr = max_v * 0.12
    in_sw, sw_start = False, 0
    windows = []
    for fi in range(n):
        if not in_sw and abs_vels[fi] >= vel_thr:
            in_sw, sw_start = True, fi
        elif in_sw and abs_vels[fi] < vel_thr * 0.25:
            if fi - sw_start >= 6:
                windows.append((sw_start, fi))
            in_sw = False
    if in_sw:
        windows.append((sw_start, n - 1))
    merged = []
    for ws, we in windows:
        if merged and ws - merged[-1][1] <= 20:
            merged[-1] = (merged[-1][0], we)
        else:
            merged.append((ws, we))

    ELBOW_MIN, LOOK_BACK = 140.0, 5
    fallback = []
    for ws, we in merged:
        ws_ext = max(0, ws - 15)
        max_ext = max(extensions[ws_ext:we + 1]) if extensions[ws_ext:we + 1] else 0
        thr = max_ext * 0.50
        best_f, best_score = we, -1.0
        for fi in range(ws_ext, we + 1):
            if extensions[fi] < thr:
                continue
            sm = all_smoothed_landmarks[fi]
            if sm is None:
                continue
            if sm[h_w].y < sm[h_s].y - 0.04:
                continue
            ea = _elbow_angle_at(all_smoothed_landmarks, fi, is_right_handed)
            if ea < ELBOW_MIN:
                continue
            prev_max = max(
                (_elbow_angle_at(all_smoothed_landmarks, pfi, is_right_handed)
                 for pfi in range(max(0, fi - LOOK_BACK), fi)),
                default=0.0
            )
            if ea >= prev_max:
                continue
            score = extensions[fi] * abs_vels[fi]
            if score > best_score:
                best_score, best_f = score, fi
        if best_score > 0:
            fallback.append(best_f)
            print(f"[✓] 팔꿈치조건 폴백: f{best_f} "
                  f"(elbow={_elbow_angle_at(all_smoothed_landmarks, best_f, is_right_handed):.1f}°, score={best_score:.5f})")

    deduped = []
    for f in sorted(fallback):
        if not deduped or f - deduped[-1] >= 150:
            deduped.append(f)

    print(f"[✓] 앙상블 최종 임팩트: {deduped}")
    return deduped


# ─────────────────────────────────────────
# 2-Pass 비디오 처리 및 렌더링 파이프라인
# ─────────────────────────────────────────

def process_video(video_url, name, is_right_handed, label=None, desc=None, speed=1.0,
                  strobe=False, strobe_frames=32, strobe_step=4, lag_scale=1.0, no_trail=False, two_handed=False,
                  audio_impact=False, ball_track=False, impact_frames=None, tracknet_model=None, ensemble=False):
    global _racket_offset_prev, _racket_face_prev, _shoe_blend_cache
    # 양손 백핸드는 와이퍼 물리가 어울리지 않으므로 lag_scale 강제 0
    if two_handed:
        lag_scale = 0.0
    _racket_offset_prev = None
    _racket_face_prev = None
    _shoe_blend_cache = {"left": None, "right": None}

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
    cap.release()

    is_serve = "serve" in name.lower()

    # 1. 미디어파이프 모델 세팅
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # ─────────────────────────────────────────
    # FIRST PASS: 생체역학 각도 및 임팩트 구간 추출 (JSON 캐싱 적용)
    # ─────────────────────────────────────────
    print("=" * 60)
    
    cache_path = f"{name}_pose_cache.json"
    loaded_from_cache = False
    all_smoothed_landmarks = []
    
    if os.path.exists(cache_path):
        import json
        print(f"[✓] Loading pose landmarks from cache: {cache_path}")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            for frame_data in cached_data:
                if frame_data is None:
                    all_smoothed_landmarks.append(None)
                else:
                    all_smoothed_landmarks.append([
                        SimpleNamespace(x=lm["x"], y=lm["y"], z=lm["z"],
                                        v=lm.get("v", 1.0))
                        for lm in frame_data
                    ])
            if len(all_smoothed_landmarks) == total_frames:
                loaded_from_cache = True
                print(f"[✓] Successfully loaded {len(all_smoothed_landmarks)} frames from cache.")
            else:
                print(f"[!] Cache frame count ({len(all_smoothed_landmarks)}) mismatch with video total_frames ({total_frames}). Re-extracting...")
                all_smoothed_landmarks = []
        except Exception as e:
            print(f"[✗] Failed to load cache: {e}. Re-extracting...")
            all_smoothed_landmarks = []

    impact_candidates = []
    
    if not loaded_from_cache:
        print("[i] Starting First Pass: Extracting pose landmarks from MediaPipe...")
        cap = cv2.VideoCapture(actual_input)
        smoother = PoseSmoother(freq=fps if fps > 0 else 30.0)
        prev_raw_landmarks = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp_ms = int(frame_idx * 1000.0 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            smoothed = None
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                # 여러 인물 중 화면 하단(hip_y 최대) 선수 선택
                best_idx = 0
                best_hip_y = -1.0
                for pi, pose_lm in enumerate(result.pose_landmarks):
                    hip_y = (pose_lm[L_HIP].y + pose_lm[R_HIP].y) / 2.0
                    if hip_y > best_hip_y:
                        best_hip_y = hip_y
                        best_idx = pi
                if best_hip_y < 0.35:
                    pass  # 상단 다른 선수만 감지됨 → 건너뜀
                else:
                    raw_landmarks = result.pose_landmarks[best_idx]
                    mutable_landmarks = [
                        SimpleNamespace(x=lm.x, y=lm.y, z=lm.z,
                                        v=lm.visibility if hasattr(lm, 'visibility') else 1.0)
                        for lm in raw_landmarks
                    ]
                    mutable_landmarks = correct_leg_swaps(mutable_landmarks, prev_raw_landmarks)
                    prev_raw_landmarks = mutable_landmarks
                    smoothed = smoother.apply(mutable_landmarks)
                
            all_smoothed_landmarks.append(smoothed)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  First Pass Processing: frame {frame_idx}/{total_frames}")
                
        cap.release()
        
        # 캐시 저장
        try:
            import json
            serialized_data = []
            for frame_data in all_smoothed_landmarks:
                if frame_data is None:
                    serialized_data.append(None)
                else:
                    serialized_data.append([
                        {"x": lm.x, "y": lm.y, "z": lm.z,
                         "v": getattr(lm, "v", 1.0)}
                        for lm in frame_data
                    ])
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(serialized_data, f, ensure_ascii=False, indent=2)
            print(f"[✓] Saved pose landmarks cache to: {cache_path}")
        except Exception as e:
            print(f"[✗] Failed to save cache: {e}")

    # 어깨 방향 기반 전역 다리 교정 (캐시 로드·신규 추출 모두 적용)
    all_smoothed_landmarks = correct_leg_orientation_global(all_smoothed_landmarks)

    # 다리 랜드마크 안정화 (hip~foot_index 인덱스 23~32)
    # 1단계: visibility < 0.5 프레임은 마지막 신뢰 값으로 대체
    # 2단계: ±3프레임 이동평균으로 잔여 노이즈 제거
    LEG_IDX = list(range(23, 33))
    VIS_THRESH = 0.5
    LEG_WIN = 3
    n_frames_total = len(all_smoothed_landmarks)

    for idx in LEG_IDX:
        # 1단계: visibility 기반 대체
        last_x, last_y = None, None
        for lm in all_smoothed_landmarks:
            if lm is None:
                continue
            vis = getattr(lm[idx], 'v', 1.0)
            if vis >= VIS_THRESH:
                last_x, last_y = lm[idx].x, lm[idx].y
            elif last_x is not None:
                lm[idx].x, lm[idx].y = last_x, last_y

        # 2단계: 이동평균 스무딩
        xs = [lm[idx].x if lm is not None else None for lm in all_smoothed_landmarks]
        ys = [lm[idx].y if lm is not None else None for lm in all_smoothed_landmarks]
        for i, lm in enumerate(all_smoothed_landmarks):
            if lm is None:
                continue
            lo, hi = max(0, i - LEG_WIN), min(n_frames_total, i + LEG_WIN + 1)
            wx = [v for v in xs[lo:hi] if v is not None]
            wy = [v for v in ys[lo:hi] if v is not None]
            if wx:
                lm[idx].x = sum(wx) / len(wx)
            if wy:
                lm[idx].y = sum(wy) / len(wy)

    # 다리 최소 간격 강제: 발목(27,28) x 차이가 MIN_FOOT_SEP 미만이면 강제 이격
    # 서브 점프 시 두 발이 2D 투영상 겹쳐 보이는 현상 방지
    L_ANKLE_IDX, R_ANKLE_IDX = 27, 28
    MIN_FOOT_SEP = 0.06   # 정규화 좌표 기준 (약 24px / 404px 캔버스)
    # 기준 방향: 과반수 프레임에서 L_ANKLE < R_ANKLE 이면 등향(L이 왼쪽)
    back_frames = sum(1 for lm in all_smoothed_landmarks
                      if lm is not None and lm[L_ANKLE_IDX].x < lm[R_ANKLE_IDX].x)
    total_valid  = sum(1 for lm in all_smoothed_landmarks if lm is not None)
    ankle_expected_back = back_frames / total_valid > 0.5

    for lm in all_smoothed_landmarks:
        if lm is None:
            continue
        la, ra = lm[L_ANKLE_IDX], lm[R_ANKLE_IDX]
        if ankle_expected_back:
            # 등향: la.x < ra.x 이어야 함
            gap = ra.x - la.x
            if gap < MIN_FOOT_SEP:
                push = (MIN_FOOT_SEP - gap) / 2
                lm[L_ANKLE_IDX].x = la.x - push
                lm[R_ANKLE_IDX].x = ra.x + push
        else:
            # 정면향: la.x > ra.x 이어야 함
            gap = la.x - ra.x
            if gap < MIN_FOOT_SEP:
                push = (MIN_FOOT_SEP - gap) / 2
                lm[L_ANKLE_IDX].x = la.x + push
                lm[R_ANKLE_IDX].x = ra.x - push

    # 임팩트 후보 분석
    impact_points = []

    if impact_frames:
        impact_points = [max(0, min(f, total_frames - 1)) for f in impact_frames]
        print(f"[✓] 수동 지정 임팩트 프레임: {impact_points}")

    if not impact_points and ball_track:
        impact_points = detect_impacts_from_tracknet(
            actual_input, fps, total_frames,
            model_path=tracknet_model or r"C:\TrackNet\model_best.pt"
        )
        if not impact_points:
            print("[!] TrackNet 공 추적 실패, 팔 신장 방식으로 대체")

    if not impact_points and audio_impact:
        # ── 오디오 + 랜드마크 이중 검증 감지 ──
        # 원리: 타구음(오디오 에너지 피크) AND 손목이 접촉존(어깨 왼쪽+허리~어깨 높이)에 있을 때만 임팩트
        audio_candidates, _ = detect_impacts_from_audio(actual_input, fps, total_frames)

        if audio_candidates and all_smoothed_landmarks:
            h_shoulder_idx = R_SHOULDER if is_right_handed else L_SHOULDER
            h_wrist_idx    = R_WRIST    if is_right_handed else L_WRIST
            h_hip_idx      = R_HIP      if is_right_handed else L_HIP
            validated = []

            h_elbow_idx = R_ELBOW if is_right_handed else L_ELBOW

            for af in audio_candidates:
                # ±10프레임 범위에서 복합 점수가 가장 높은 프레임 선택
                best_f, best_score = af, -1
                best_debug = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                for fi in range(max(0, af - 10), min(total_frames, af + 10)):
                    sm = all_smoothed_landmarks[fi]
                    if sm is None:
                        continue
                    s = sm[h_shoulder_idx]
                    w = sm[h_wrist_idx]
                    h = sm[h_hip_idx]
                    e = sm[h_elbow_idx]

                    # 1차 필터: 손목이 어깨에서 수평으로 0.03 이상 떨어짐(방향 무관)
                    #            + 높이: 어깨 위 0.05 ~ 어깨 아래 0.30 범위 (낮은 자세 허용)
                    wrist_away_from_shoulder = abs(w.x - s.x) > 0.03
                    wrist_in_height_zone     = (s.y - 0.05 <= w.y <= s.y + 0.30)
                    if not (wrist_away_from_shoulder and wrist_in_height_zone):
                        continue

                    # 하드 필터: elbow 각도 120° 미만은 팔꿈치 과굴곡 → 임팩트 아님
                    elbow_angle = calculate_angle_2d((s.x, s.y), (e.x, e.y), (w.x, w.y))
                    if elbow_angle < 120.0:
                        continue

                    # 팔 신장 및 x오프셋 (방향 무관 절댓값)
                    arm_extension = math.sqrt((w.x - s.x)**2 + (w.y - s.y)**2)
                    x_offset = abs(w.x - s.x)

                    # 소프트 점수: 알카라스 포핸드 임팩트 통계 기반 (0~1, 가까울수록 높음)
                    # x오프셋 피크 0.135 (실측 범위 0.10~0.17)
                    x_score   = max(0.0, 1.0 - abs(x_offset      - 0.135) / 0.135)
                    # 팔 신장 피크 0.140 (실측 범위 0.10~0.18)
                    ext_score = max(0.0, 1.0 - abs(arm_extension  - 0.140) / 0.100)
                    # elbow 각도 피크 156° (실측 범위 140~170°)
                    ang_score = max(0.0, 1.0 - abs(elbow_angle    - 156.0) /  30.0)

                    # 복합 점수: x오프셋(주 기준) + 소프트 가중치
                    score = x_offset + (x_score + ext_score + ang_score) * 0.1

                    if score > best_score:
                        best_score, best_f = score, fi
                        best_debug = (elbow_angle, arm_extension, x_offset, x_score, ext_score, ang_score)

                if best_score > 0:
                    validated.append(best_f)
                    ea, ae, xo, xs, es, ags = best_debug
                    print(f"[✓] Validated impact: f{best_f} (audio={af}, score={best_score:.3f} | "
                          f"elbow={ea:.0f}° ext={ae:.3f} xoff={xo:.3f} | "
                          f"scores: x={xs:.2f} ext={es:.2f} ang={ags:.2f})")
                else:
                    print(f"[i] Audio peak f{af} rejected: wrist not in contact zone")

            # 중복 제거 (150프레임 이내)
            impact_points = []
            for f in sorted(validated):
                if not impact_points or f - impact_points[-1] >= 150:
                    impact_points.append(f)

        if not impact_points:
            print("[!] 오디오+포지션 검증 실패, 팔 신장 방식으로 대체")

    if not impact_points and not is_serve:
        if ensemble:
            impact_points = detect_impacts_ensemble(
                actual_input, fps, total_frames, all_smoothed_landmarks, is_right_handed
            )
            if not impact_points:
                print("[!] 앙상블 감지 실패, 팔신장 단독으로 대체")
                impact_points, _ = detect_impacts_arm_extension(all_smoothed_landmarks, is_right_handed)
        else:
            impact_points, _ = detect_impacts_arm_extension(all_smoothed_landmarks, is_right_handed)

    elif is_serve:
        # 서브: 기존 phase 기반 감지
        for frame_idx, smoothed in enumerate(all_smoothed_landmarks):
            if smoothed is not None:
                h_shoulder = smoothed[R_SHOULDER] if is_right_handed else smoothed[L_SHOULDER]
                h_elbow    = smoothed[R_ELBOW]    if is_right_handed else smoothed[L_ELBOW]
                h_wrist    = smoothed[R_WRIST]    if is_right_handed else smoothed[L_WRIST]
                elbow_angle = calculate_angle_2d(
                    (h_shoulder.x, h_shoulder.y),
                    (h_elbow.x, h_elbow.y),
                    (h_wrist.x, h_wrist.y)
                )
                phase = detect_serve_phase(smoothed, is_right_handed, elbow_angle)
                if phase == "Impact":
                    impact_candidates.append(frame_idx)

        if impact_candidates:
            groups = []
            current_group = [impact_candidates[0]]
            for val in impact_candidates[1:]:
                if val == current_group[-1] + 1:
                    current_group.append(val)
                else:
                    groups.append(current_group)
                    current_group = [val]
            groups.append(current_group)
            for g in groups:
                mid_f = g[len(g) // 2]
                shifted_mid_f = max(0, mid_f - 3)
                impact_points.append(shifted_mid_f)
                print(f"[✓] Bio-impact detected at Frame {shifted_mid_f} (originally {mid_f})")
    if not impact_points:
        # 임팩트 미검출 시 폴백
        mid_fallback = total_frames // 2
        impact_points.append(mid_fallback)
        print(f"[i] Fallback: Using midpoint frame {mid_fallback} as impact point.")

    # 서브 동작 시 중복 감지된 오검출 포인트 제거 필터링
    if is_serve and len(impact_points) > 1:
        best_imp = None
        min_y = 9999.0
        h_wrist_idx = R_WRIST if is_right_handed else L_WRIST
        for imp_f in impact_points:
            smoothed = all_smoothed_landmarks[imp_f]
            if smoothed is not None:
                y_val = smoothed[h_wrist_idx].y
                if y_val < min_y:
                    min_y = y_val
                    best_imp = imp_f
        if best_imp is not None:
            print(f"[i] Filtered duplicate serve impacts. Kept Frame {best_imp} (wrist.y={min_y:.4f}) and discarded others: {[x for x in impact_points if x != best_imp]}")
            impact_points = [best_imp]

    # 지상타격 속도 필터: 팔 신장 방식으로 전환 후 불필요 → 서브에만 적용
    if False and not is_serve and not two_handed and len(impact_points) > 0:
        filtered_points = []
        h_wrist_idx = R_WRIST if is_right_handed else L_WRIST
        for imp_f in impact_points:
            # 전후 4프레임 간의 평균 손목 속도 계산
            if imp_f - 2 >= 0 and imp_f + 2 < len(all_smoothed_landmarks):
                lm_prev = all_smoothed_landmarks[imp_f - 2]
                lm_next = all_smoothed_landmarks[imp_f + 2]
                if lm_prev is not None and lm_next is not None:
                    w_prev = lm_prev[h_wrist_idx]
                    w_next = lm_next[h_wrist_idx]
                    wrist_vel = math.sqrt((w_next.x - w_prev.x)**2 + (w_next.y - w_prev.y)**2) / 4.0
                    # 속도가 0.0028 이상인 진짜 스윙 가속 타격 프레임만 유지
                    if wrist_vel >= 0.0028:
                        filtered_points.append(imp_f)
                    else:
                        print(f"[i] Filtered slow groundstroke impact: Frame {imp_f} (speed={wrist_vel:.6f})")
                else:
                    filtered_points.append(imp_f)
            else:
                filtered_points.append(imp_f)
        impact_points = filtered_points

    # 전체 공통: 너무 가까운 임팩트 중복 제거 (150프레임 이내 = 5초 이내)
    # 과다 감지 방지 — 포핸드/백핸드 모두 적용
    if not is_serve and len(impact_points) > 1:
        deduped = []
        for imp in sorted(impact_points):
            if deduped and imp - deduped[-1] < int(fps * 4):  # 4초 미만만 중복 제거
                pass  # 너무 가까우면 첫 번째 유지(이미 속도 필터 통과한 것)
            else:
                deduped.append(imp)
        if len(deduped) < len(impact_points):
            print(f"[i] Deduped impacts: {len(impact_points)} → {len(deduped)}")
        impact_points = deduped

    # ─────────────────────────────────────────
    # SECOND PASS: 잔상 오버레이 렌더링 및 비디오 쓰기
    # ─────────────────────────────────────────
    print("=" * 60)
    print("[i] Starting Second Pass: Rendering stickman with strobe effect...")
    cap = cv2.VideoCapture(actual_input)

    out_h = OUT_H
    out_w = int(round(orig_w * out_h / orig_h / 2)) * 2
    render_w, render_h = out_w * SSAA, out_h * SSAA
    configure_thickness(render_h)

    court_bg = draw_court_background(render_w, render_h)

    # cv2 VideoWriter 대신 ffmpeg stdin 파이프로 직접 인코딩 (Windows mp4v FPS 버그 우회)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    out = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    frame_idx = 0
    phase_smoother = PhaseSmoother()
    strobe_history = [] if strobe else None
    
    racket_trail = [] if not no_trail else None
    hand_trail = [] if not no_trail else None

    # 재생 속도 제어용 프레임 보간 카운터
    frame_multiplier = 1.0 / speed
    accumulated_frames = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 가장 가까운 임팩트 지점과의 프레임 거리 k 계산
        k = 9999
        if impact_points:
            nearest_imp = min(impact_points, key=lambda x: abs(frame_idx - x))
            k = frame_idx - nearest_imp

        canvas = court_bg.copy()
        current_phase = "Preparation"

        smoothed = all_smoothed_landmarks[frame_idx]
        if smoothed is not None:
            current_phase = draw_stickman(
                canvas, smoothed, render_w, render_h, is_right_handed,
                racket_trail=racket_trail, hand_trail=hand_trail, phase_smoother=phase_smoother, is_serve=is_serve,
                k=k, strobe_history=strobe_history, strobe_frames=strobe_frames, strobe_step=strobe_step,
                lag_scale=lag_scale, two_handed=two_handed
            )

        final = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)
        
        # 손목 속도벡터 화살표 (속도 크기에 따라 녹→적 색상)
        if smoothed is not None:
            look = min(frame_idx + 3, len(all_smoothed_landmarks) - 1)
            next_sm = all_smoothed_landmarks[look]
            if next_sm is not None:
                h_wi = R_WRIST if is_right_handed else L_WRIST
                wx0 = int(smoothed[h_wi].x * out_w)
                wy0 = int(smoothed[h_wi].y * out_h)
                wx1 = int(next_sm[h_wi].x * out_w)
                wy1 = int(next_sm[h_wi].y * out_h)
                vel = math.sqrt((wx1 - wx0) ** 2 + (wy1 - wy0) ** 2)
                if vel > 3:
                    scale = min(6.0, max(1.5, vel / 4.0))
                    tip_x = int(wx0 + (wx1 - wx0) * scale)
                    tip_y = int(wy0 + (wy1 - wy0) * scale)
                    spd_norm = min(1.0, vel / 28.0)
                    vec_color = (0, int(255 * (1 - spd_norm)), int(255 * spd_norm))
                    thick = max(2, int(2 + 2 * spd_norm))
                    cv2.arrowedLine(final, (wx0, wy0), (tip_x, tip_y),
                                    vec_color, thick, cv2.LINE_AA, tipLength=0.25)

        # 임팩트 플래시 효과 (k=0~8 구간, 흰색 페이드아웃)
        for imp_f in impact_points:
            frames_since = frame_idx - imp_f
            if 0 <= frames_since <= 8:
                alpha = max(0.0, 0.55 * (1.0 - frames_since / 8.0))
                flash = np.full_like(final, 255)
                cv2.addWeighted(flash, alpha, final, 1.0 - alpha, 0, final)

        # 임팩트 텍스트 (IMPACT!)
        for imp_f in impact_points:
            if -2 <= frame_idx - imp_f < 20:
                cv2.putText(final, "IMPACT!", (out_w // 2 - 160, out_h - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (20, 20, 20), 8, cv2.LINE_AA)
                cv2.putText(final, "IMPACT!", (out_w // 2 - 160, out_h - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 255, 50), 4, cv2.LINE_AA)

        if label:
            final = draw_label(final, label, desc)
            
        accumulated_frames += frame_multiplier
        write_count = int(accumulated_frames)
        accumulated_frames -= write_count
        for _ in range(write_count):
            out.stdin.write(final.tobytes())

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"  Rendering: frame {frame_idx}/{total_frames} ({pct:.1f}%)")

    cap.release()
    out.stdin.close()
    out.wait()
    landmarker.close()

    # 원본 오디오 합성
    print("[i] 원본 오디오 합성 중...")
    audio_temp = output_path.replace(".mp4", "_audiomerge.mp4")
    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", output_path,
            "-i", actual_input,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            audio_temp
        ], capture_output=True)
        if result.returncode == 0:
            # Windows에서 파일이 잠겨 있으면 직접 교체 대신 별도 파일로 저장
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(audio_temp, output_path)
                print("[✓] 원본 오디오 합성 완료")
            except OSError:
                # 원본 파일이 열려 있으면 _with_audio 접미사로 저장
                alt_path = output_path.replace(".mp4", "_with_audio.mp4")
                if os.path.exists(alt_path):
                    os.remove(alt_path)
                os.rename(audio_temp, alt_path)
                print(f"[✓] 원본 오디오 합성 완료 (파일 잠김으로 별도 저장): {alt_path}")
        else:
            print("[!] 오디오 합성 실패 (원본에 오디오 없음?), 무음으로 저장")
            if os.path.exists(audio_temp):
                os.remove(audio_temp)
    except Exception as e:
        print(f"[!] 오디오 합성 오류: {e}")

    print(f"\n[✓] Stickman video saved: {output_path}")
    print(f"    Resolution: {out_w}x{out_h} @ {fps:.1f}fps, {frame_idx} frames")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"  TENNIS STICKMAN ANIMATION GENERATOR v12.4")
    print(f"  동작: {args.name}  |  손: {'왼손' if args.left else '오른손'}  |  배속: {args.speed}x  |  잔상: {'ON' if args.strobe else 'OFF'}  |  앙상블: {'ON' if args.ensemble else 'OFF'}")
    print("=" * 60)
    process_video(args.url, args.name, is_right_handed=not args.left,
                  label=args.label, desc=args.desc, speed=args.speed,
                  strobe=args.strobe, strobe_frames=args.strobe_frames, strobe_step=args.strobe_step,
                  lag_scale=args.lag_scale, no_trail=args.no_trail, two_handed=args.two_handed,
                  audio_impact=args.audio_impact, ball_track=args.ball_track,
                  impact_frames=args.impact_frame, tracknet_model=args.tracknet_model,
                  ensemble=args.ensemble)

