# 테니스 동작 영상을 스틱맨 애니메이션으로 변환하는 생성기 (v5: 가는 팔다리·목·다듬은 신발)
"""
Tennis Stickman Animation Generator v5 (인간 인지 품질 향상 버전)
사용법:
  python tennis_stickman_v5.py <YouTube_URL_또는_로컬파일> <동작명> [--left] [--speed <배속>]

예시:
  python tennis_stickman_v5.py C:/Users/.../federer_input.mp4 federer_serve --speed 0.5

v4 및 이전 버전 대비 변경 사항 (인간의 인지적 수용성을 높이기 위한 HUD 디자인 고도화):
  1. 각도 기호 깨짐 해결: OpenCV 폰트 대신 PIL을 이용해 유니코드 기호(°)가 깨짐 없이 깔끔하게 출력되도록 일괄 개선.
  2. 스틱맨 체형 및 의복 보완: 흰색 상의(몸통 채우기)와 다각형 기반 흰색 반바지(Shorts)를 적용하여 인체의 비례와 신체 방향성의 시각 단서 보완.
  3. 3차원 입체 레이어링 최적화: 뒤쪽 팔(토스팔)이 얼굴을 관통하지 않고 머리와 몸통 뒤로 깔끔하게 숨겨지도록 그리기 순서(마스킹) 재설정.
  4. 시각 노이즈 억제: 관절 각도 오버레이 아크(Arc) 반경을 콤팩트하게 축소(13 * LW)하여 겹침을 방지하고 가독성 확보.
  5. 배속 필터 적용 (--speed): 프레임 복제 기반의 부드러운 슬로우 모션을 제공하여 인간 눈의 시간적 인지 한계를 극복.
  6. 모션 궤적(Neon Trails): 라켓 헤드와 손목 궤적에 네온 광원 성향의 긴 페이딩 잔상을 적용하여 스윙의 흐름과 속도를 시각화.
  7. 동작 구간 배지(Serve Phase): 트로피 포즈, 라켓 드롭, 임팩트, 팔로우 스루 등 구간 자동 판정 및 우측 상단 네온 배지 표시.
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
        description="테니스 스틱맨 애니메이션 생성기 v5 (각도·궤적·구간 자동분석 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python tennis_stickman_v5.py https://youtu.be/xxxx federer_serve
  python tennis_stickman_v5.py https://youtu.be/xxxx nadal_forehand --left --speed 0.5
  python tennis_stickman_v5.py https://youtu.be/xxxx djokovic_backhand
        """
    )
    parser.add_argument("url",  help="YouTube 영상 URL 또는 로컬 파일 경로")
    parser.add_argument("name", help="동작명 (파일명에 사용, 예: federer_serve, nadal_forehand)")
    parser.add_argument("--left", action="store_true", help="왼손잡이 선수 (기본: 오른손잡이)")
    parser.add_argument("--label", default=None, help="화면에 표시할 동작 이름 자막 (예: 포핸드)")
    parser.add_argument("--desc",  default=None, help="자막 아래 줄 설명 (선택)")
    parser.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율 (예: 0.5는 2배 느린 슬로우 모션)")
    return parser.parse_args()


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker_full.task"

SSAA  = 2      # 슈퍼샘플링 배수 (렌더 = 출력 × SSAA, 다운스케일로 선 선명)
OUT_H = 720    # 출력 세로 해상도 (입력 360 → 업스케일, 벡터처럼 선명). 360 기준 두께 비율 보존

# ── 선 두께 단위 LW ──
#  렌더 높이 720(=360×SSAA)에서 LW=SSAA가 되도록 정의 → 360p와 동일한 두께 비율을 어떤 해상도에서도 유지.
#  몸 비례 크기(head_r·라켓·신발)는 figure 크기에 자동 비례하므로 LW와 무관.
LW = SSAA

def configure_thickness(render_h):
    """렌더 높이에 맞춰 선 두께 단위와 명명된 두께 상수를 설정."""
    global LW, LIMB_THICKNESS, NECK_THICKNESS, HEAD_OUTLINE_THICKNESS, TORSO_OUTLINE_THICKNESS
    LW = SSAA * (render_h / 720.0)
    LIMB_THICKNESS          = max(int(6 * LW), 2)
    NECK_THICKNESS          = max(int(7 * LW), 2)
    HEAD_OUTLINE_THICKNESS  = max(int(8 * LW), 2)
    TORSO_OUTLINE_THICKNESS = max(int(6 * LW), 2)

BODY_COLOR        = (20, 20, 20)
LIMB_THICKNESS    = 6 * SSAA    # 팔·다리 (가늘게) — configure_thickness로 재설정됨
NECK_THICKNESS    = 7 * SSAA    # 목
HEAD_OUTLINE_THICKNESS  = 8 * SSAA   # 머리 외곽선 (제일 굵게)
TORSO_OUTLINE_THICKNESS = 6 * SSAA   # 몸통 외곽선 (팔다리 수준, 속은 비움)
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

# ── 모션 보간(스무딩) ──
#  One-Euro 필터: 정지 시엔 강하게 평활(떨림 제거), 빠른 동작 시엔 컷오프를 올려 반응성 유지(지연 최소화).
ONE_EURO_MIN_CUTOFF = 1.0    # 낮을수록 정지 시 더 부드러움(지연↑)
ONE_EURO_BETA       = 0.7    # 높을수록 빠른 동작에 더 민감(지연↓)
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
    """1차원 One-Euro 저역통과 필터. 속도에 따라 컷오프를 적응적으로 조절."""
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
    """랜드마크 33개 × (x,y,z)를 각각 One-Euro로 평활."""
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
    """출력 해상도 프레임에 동작 이름(label)과 설명(desc)을 반투명 바와 함께 그림. 한글 지원(PIL)."""
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
# Step 3: Stickman Drawing
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
    # 1. 내부를 흰색으로 채움 (셔츠 효과 - 일반 다각형 대응을 위해 fillPoly 사용)
    cv2.fillPoly(canvas, [pts], (255, 255, 255), cv2.LINE_AA)
    # 2. 외곽선 그리기
    cv2.polylines(canvas, [pts], isClosed=True, color=OUTLINE_COLOR,
                  thickness=TORSO_OUTLINE_THICKNESS, lineType=cv2.LINE_AA)


def draw_hand(canvas, wrist, radius):
    cv2.circle(canvas, wrist, radius, HAND_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, wrist, radius, HAND_OUTLINE_COLOR, max(int(3 * LW), 3), cv2.LINE_AA)


def draw_shoe(canvas, ankle, heel, toe, knee, size, is_back_view):
    """발 방향(toe-heel)의 2D 정규화 길이에 따라 원근 형태를 결정합니다.
    - 정면/후면 뷰 (L <= size * 0.45): 발이 단축(Foreshortening)되므로 발목 기준 대칭형 신발로 렌더링.
      - is_back_view 가 True 인 경우: 백뷰 (Heel View, 힐 탭 스트립 표시)
      - is_back_view 가 False 인 경우: 프론트 뷰 (Toe View, 흰색 신발끈 3줄, 토캡 아크 표시)
    - 측면 뷰 (L > size * 0.45): 비대칭 로퍼형 측면 프로파일로 렌더링.
    """
    ankle = np.array(ankle, dtype=np.float64)
    heel  = np.array(heel,  dtype=np.float64)
    toe   = np.array(toe,   dtype=np.float64)
    knee  = np.array(knee,  dtype=np.float64)

    foot = toe - heel
    L = np.linalg.norm(foot)

    # 1. 정면/후면 대칭형 신발 렌더링 (단축된 발)
    if L <= size * 0.45:
        v = ankle - knee
        d_dir = v / (np.linalg.norm(v) + 1e-6)  # 정강이 방향 (아래쪽)
        w_dir = np.array([-d_dir[1], d_dir[0]])  # 수평 폭 방향

        if is_back_view:
            # 후면 시점 (Heel View)
            profile = [
                (-0.22, 0.00),  # ankle left
                (-0.32, 0.20),  # heel mid-left
                (-0.40, 0.60),  # sole left corner
                (-0.20, 0.65),  # sole inner-left
                ( 0.20, 0.65),  # sole inner-right
                ( 0.40, 0.60),  # sole right corner
                ( 0.32, 0.20),  # heel mid-right
                ( 0.22, 0.00),  # ankle right
            ]
            pts = np.array([
                (ankle + w_dir * fx * size + d_dir * fy * size)
                for fx, fy in profile
            ], dtype=np.int32)

            cv2.fillPoly(canvas, [pts], SHOE_FILL_COLOR, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                          thickness=max(int(3 * LW), 2), lineType=cv2.LINE_AA)

            # 밑창
            sole = np.array([
                (ankle - w_dir * (size * 0.40) + d_dir * (size * 0.60)),
                (ankle - w_dir * (size * 0.20) + d_dir * (size * 0.65)),
                (ankle + w_dir * (size * 0.20) + d_dir * (size * 0.65)),
                (ankle + w_dir * (size * 0.40) + d_dir * (size * 0.60))
            ], dtype=np.int32)
            cv2.polylines(canvas, [sole], isClosed=False, color=(110, 110, 110),
                          thickness=max(int(2 * LW), 2), lineType=cv2.LINE_AA)

            # 세로형 힐 탭 스트립 라인 (짧게 조정하여 다리 스틱이 내려온 착시 해결)
            heel_strip_start = (ankle + d_dir * (size * 0.05)).astype(np.int32)
            heel_strip_end = (ankle + d_dir * (size * 0.22)).astype(np.int32)
            cv2.line(canvas, heel_strip_start, heel_strip_end, SHOE_OUTLINE_COLOR,
                     max(int(2.5 * LW), 2), cv2.LINE_AA)
        else:
            # 전면 시점 (Toe View)
            profile = [
                (-0.22, 0.00),  # ankle left
                (-0.35, 0.25),  # mid-foot left
                (-0.42, 0.70),  # sole left corner
                (-0.20, 0.76),  # toe cap left
                ( 0.20, 0.76),  # toe cap right
                ( 0.42, 0.70),  # sole right corner
                ( 0.35, 0.25),  # mid-foot right
                ( 0.22, 0.00),  # ankle right
            ]
            pts = np.array([
                (ankle + w_dir * fx * size + d_dir * fy * size)
                for fx, fy in profile
            ], dtype=np.int32)

            cv2.fillPoly(canvas, [pts], SHOE_FILL_COLOR, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                          thickness=max(int(3 * LW), 2), lineType=cv2.LINE_AA)

            # 밑창
            sole = np.array([
                (ankle - w_dir * (size * 0.42) + d_dir * (size * 0.70)),
                (ankle - w_dir * (size * 0.20) + d_dir * (size * 0.76)),
                (ankle + w_dir * (size * 0.20) + d_dir * (size * 0.76)),
                (ankle + w_dir * (size * 0.42) + d_dir * (size * 0.70))
            ], dtype=np.int32)
            cv2.polylines(canvas, [sole], isClosed=False, color=(110, 110, 110),
                          thickness=max(int(2 * LW), 2), lineType=cv2.LINE_AA)

            # 세로형 신발끈 가이드 라인
            lace_start = (ankle + d_dir * (size * 0.12))
            lace_end = (ankle + d_dir * (size * 0.45))
            cv2.line(canvas, lace_start.astype(np.int32), lace_end.astype(np.int32),
                     SHOE_OUTLINE_COLOR, max(int(1 * LW), 1), cv2.LINE_AA)

            # 가로형 흰색 신발끈 3줄
            for frac in [0.20, 0.30, 0.40]:
                bar_center = ankle + d_dir * (size * frac)
                bar_left = (bar_center - w_dir * (size * 0.12)).astype(np.int32)
                bar_right = (bar_center + w_dir * (size * 0.12)).astype(np.int32)
                cv2.line(canvas, bar_left, bar_right, (255, 255, 255),
                         max(int(1 * LW), 1), cv2.LINE_AA)

            # 토캡 라인 (경계선)
            toe_cap_center = ankle + d_dir * (size * 0.52)
            toe_cap_left = (toe_cap_center - w_dir * (size * 0.32)).astype(np.int32)
            toe_cap_right = (toe_cap_center + w_dir * (size * 0.32)).astype(np.int32)
            cv2.line(canvas, toe_cap_left, toe_cap_right, SHOE_OUTLINE_COLOR,
                     max(int(1.2 * LW), 2), cv2.LINE_AA)

    # 2. 측면 비대칭 로퍼형 신발 렌더링
    else:
        u = foot / L
        perp = np.array([-u[1], u[0]])
        if perp[1] < 0:          # 항상 아래(지면)를 향하도록
            perp = -perp

        length = size * 1.15
        height = size * 0.62

        profile = [
            (-0.12, 0.00),   # 뒤축 위 (다리 만나는 지점)
            (-0.22, 0.20),   # 뒤꿈치 뒤
            (-0.24, 0.50),   # 뒤꿈치 곡선
            (-0.18, 0.78),   # 뒤꿈치 아래
            (-0.06, 0.94),   # 밑창 뒤
            ( 0.18, 1.00),   # 밑창
            ( 0.48, 1.00),   # 밑창
            ( 0.72, 0.97),   # 밑창 앞
            ( 0.90, 0.88),   # 앞코 아래
            ( 1.00, 0.72),   # 앞코 앞
            ( 1.02, 0.54),   # 앞코 끝 (둥글고 뭉툭)
            ( 0.96, 0.36),   # 앞코 위
            ( 0.82, 0.24),   # 앞코 위 (높게 유지 → 통통)
            ( 0.60, 0.16),   # 발등
            ( 0.38, 0.11),   # 발등
            ( 0.18, 0.06),   # 입구 앞
        ]
        pts = np.array([
            (ankle + u * fx * length + perp * fy * height)
            for fx, fy in profile
        ], dtype=np.int32)

        cv2.fillPoly(canvas, [pts], SHOE_FILL_COLOR, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                      thickness=max(int(3 * LW), 2), lineType=cv2.LINE_AA)

        # 밑창 (진한 바닥 라인)
        sole = np.array([
            (ankle + u * fx * length + perp * fy * height)
            for fx, fy in [(-0.06, 0.94), (0.18, 1.00), (0.48, 1.00), (0.72, 0.97)]
        ], dtype=np.int32)
        cv2.polylines(canvas, [sole], isClosed=False, color=(110, 110, 110),
                      thickness=max(int(2 * LW), 2), lineType=cv2.LINE_AA)


def draw_shadow(canvas, l_ankle, r_ankle):
    cx = (l_ankle[0] + r_ankle[0]) // 2
    cy = max(l_ankle[1], r_ankle[1]) + int(6 * LW)
    spread = max(abs(l_ankle[0] - r_ankle[0]), int(40 * LW))
    overlay = canvas.copy()
    cv2.ellipse(overlay, (cx, cy), (int(spread * 0.7), max(int(spread * 0.1), int(6 * LW))),
                0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)


def draw_racket(canvas, wrist, elbow, head_r):
    wx, wy = wrist
    ex, ey = elbow
    dx, dy = wx - ex, wy - ey
    arm_len = math.sqrt(dx * dx + dy * dy) + 1e-6
    nx, ny = dx / arm_len, dy / arm_len

    grip_length = int(head_r * 1.2)
    frame_rx = int(head_r * 1.15)
    frame_ry = int(head_r * 1.5)

    grip_end_x = int(wx + nx * grip_length)
    grip_end_y = int(wy + ny * grip_length)
    cv2.line(canvas, (wx, wy), (grip_end_x, grip_end_y),
             RACKET_GRIP_COLOR, max(int(8 * LW), 6), cv2.LINE_AA)

    head_cx = int(grip_end_x + nx * frame_ry)
    head_cy = int(grip_end_y + ny * frame_ry)
    angle = math.degrees(math.atan2(ny, nx))

    cv2.ellipse(canvas, (head_cx, head_cy), (frame_rx, frame_ry),
                angle, 0, 360, RACKET_FRAME_COLOR, max(int(7 * LW), 5), cv2.LINE_AA)

    string_thick = max(1, int(round(LW)))
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


# ─────────────────────────────────────────
# 생체역학 분석 및 시각화 도구 (각도, 궤적, 동작 분석)
# ─────────────────────────────────────────

class PhaseSmoother:
    """프레임 간 동작 구간 이름의 흔들림(Flickering)을 방지하기 위한 필터."""
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.history = []
        
    def add_and_get(self, phase):
        self.history.append(phase)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return max(set(self.history), key=self.history.count)


def calculate_angle_2d(p1, p2, p3):
    """2차원 화면 좌표 기준 세 점(p2가 꼭짓점) 사이의 각도를 반환."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def draw_angle_overlay(canvas, joint, p1, p2, color):
    """지정된 관절에 반투명한 각도 호(arc)를 그리고, 각도 텍스트 그리기 태스크를 반환."""
    angle_val = calculate_angle_2d(p1, joint, p2)
    
    # 벡터 방향 계산
    v1 = np.array(p1) - np.array(joint)
    v2 = np.array(p2) - np.array(joint)
    
    # 극좌표계 상의 각도(라디안) 구하기
    ang1 = math.atan2(v1[1], v1[0])
    ang2 = math.atan2(v2[1], v2[0])
    
    # 도 단위 변환
    deg1 = int(np.degrees(ang1))
    deg2 = int(np.degrees(ang2))
    
    # 시계방향/반시계방향 중 최소 각도 호 선택
    diff = (deg2 - deg1) % 360
    if diff > 180:
        start_angle = deg2
        end_angle = deg1 + 360
    else:
        start_angle = deg1
        end_angle = deg2
        
    overlay = canvas.copy()
    radius = int(13 * LW) # 호의 반경 축소 (프리뷰 크기 매칭)
    
    # 1. 반투명한 부채꼴 채우기
    cv2.ellipse(overlay, joint, (radius, radius), 0, start_angle, end_angle, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
    
    # 2. 부채꼴 테두리 그리기
    cv2.ellipse(canvas, joint, (radius, radius), 0, start_angle, end_angle, color, max(1, int(1.2 * LW)), cv2.LINE_AA)
    
    # 3. 각도 텍스트 표시 위치 계산 (이등분선 방향으로 마진 부여)
    bisector_deg = (start_angle + end_angle) / 2
    bisector_rad = np.radians(bisector_deg)
    
    text_dist = radius + int(8 * LW)
    tx = int(joint[0] + text_dist * np.cos(bisector_rad))
    ty = int(joint[1] + text_dist * np.sin(bisector_rad))
    
    text = f"{int(round(angle_val))}°"
    
    # BGR 색상을 PIL에서 사용할 RGB 색상으로 변환하여 반환
    rgb_color = (color[2], color[1], color[0])
    return (text, (tx, ty), rgb_color)


def draw_texts_pil(canvas, tasks, font_size):
    """여러 텍스트 그리기 태스크를 하나의 PIL 컨텍스트에서 고성능으로 일괄 처리 (유니코드 기호 완벽 지원)."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    
    font = ImageFont.truetype(FONT_PATH, int(round(font_size)))
    
    for text, (x, y), color in tasks:
        tb = draw.textbbox((0, 0), text, font=font)
        # 1. 가독성을 위한 검은색 아웃라인
        outline_color = (20, 20, 20, 255)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy - tb[1]), text, font=font, fill=outline_color)
        # 2. 본문 텍스트
        draw.text((x, y - tb[1]), text, font=font, fill=color + (255,))
        
    canvas_new = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    np.copyto(canvas, canvas_new)


def draw_glowing_trail(canvas, trail, color):
    """부드럽게 페이드아웃되는 네온 글로우(Glow) 궤적을 렌더링."""
    if len(trail) < 2:
        return
    n = len(trail)
    
    # 1. 외부 광원(Glow) 렌더링
    overlay = canvas.copy()
    for i in range(1, n):
        t = i / (n - 1)
        thick = max(1, int(10 * LW * t))
        cv2.line(overlay, trail[i-1], trail[i], color, thick, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    
    # 2. 중심부 흰색 코어(Core) 렌더링
    overlay_core = canvas.copy()
    for i in range(1, n):
        t = i / (n - 1)
        thick = max(1, int(3 * LW * t))
        cv2.line(overlay_core, trail[i-1], trail[i], (255, 255, 255), thick, cv2.LINE_AA)
    cv2.addWeighted(overlay_core, 0.65, canvas, 0.35, 0, canvas)


def detect_serve_phase(lm, is_right_handed, elbow_angle):
    """미디어파이프 정규화 좌표를 분석해 서브 시퀀스의 단계를 감지."""
    h_wrist = lm[R_WRIST] if is_right_handed else lm[L_WRIST]
    h_shoulder = lm[R_SHOULDER] if is_right_handed else lm[L_SHOULDER]
    h_elbow = lm[R_ELBOW] if is_right_handed else lm[L_ELBOW]
    
    nh_wrist = lm[L_WRIST] if is_right_handed else lm[R_WRIST]
    nh_shoulder = lm[L_SHOULDER] if is_right_handed else lm[R_SHOULDER]
    
    nose = lm[NOSE]
    
    # 1. 임팩트 (Impact)
    # 치는 손목이 코보다 높고 팔꿈치 각도가 거의 일직선(155도 이상)
    if h_wrist.y < nose.y and elbow_angle > 155:
        return "Impact"
        
    # 2. 라켓 드롭 (Racket Drop)
    # 치는 팔꿈치가 어깨선 근처로 올라오고, 팔꿈치는 깊게 굽혀져 있으며, 손목이 팔꿈치보다 낮음 (라켓이 등 뒤로 떨어진 상태)
    if h_elbow.y < h_shoulder.y + 0.05 and elbow_angle < 95 and h_wrist.y > h_elbow.y:
        return "Racket Drop"
        
    # 3. 트로피 포즈 (Trophy Pose)
    # 반대 손(토스한 손)이 어깨보다 높고, 치는 팔꿈치는 준비 동작으로 굽혀짐(60~130도 사이)
    if nh_wrist.y < nh_shoulder.y - 0.05 and elbow_angle < 130 and elbow_angle > 60:
        return "Trophy Pose"
        
    # 4. 팔로우 스루 (Follow Through)
    # 스윙 후 치는 손이 어깨 아래로 내려가고 반대편 어깨 방향으로 가로지름
    if is_right_handed:
        if h_wrist.y > h_shoulder.y and h_wrist.x < lm[L_SHOULDER].x:
            return "Follow Through"
    else:
        if h_wrist.y > h_shoulder.y and h_wrist.x > lm[R_SHOULDER].x:
            return "Follow Through"
            
    # 5. 준비 (Preparation)
    return "Preparation"


def detect_groundstroke_phase(lm, is_right_handed, elbow_angle):
    """미디어파이프 정규화 좌표를 분석해 그라운드스트로크(포핸드/백핸드) 시퀀스의 단계를 감지."""
    h_wrist = lm[R_WRIST] if is_right_handed else lm[L_WRIST]
    h_shoulder = lm[R_SHOULDER] if is_right_handed else lm[L_SHOULDER]
    nh_shoulder = lm[L_SHOULDER] if is_right_handed else lm[R_SHOULDER]
    h_hip = lm[R_HIP] if is_right_handed else lm[L_HIP]
    
    # 2D 평면 거리 계산
    def dist_2d(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
    d_opp_shoulder = dist_2d(h_wrist, nh_shoulder)
    
    # 1. 팔로우 스루 (Follow Through): 치는 손목이 반대편 어깨와 매우 가까움
    if d_opp_shoulder < 0.18 and h_wrist.y < h_shoulder.y + 0.05:
        return "Follow Through"
        
    # 2. 임팩트 (Impact): 팔이 어느 정도 펴져 있고(각도 > 135) 손목 높이가 몸통 영역 내에 있음
    if elbow_angle > 135 and h_wrist.y > h_shoulder.y - 0.05 and h_wrist.y < h_hip.y + 0.1:
        return "Impact"
        
    # 3. 테이크 백 (Take Back): 손이 어깨와 골반보다 훨씬 멀리 떨어짐 (테이크백 깊이)
    if d_opp_shoulder > 0.35:
        return "Take Back"
        
    # 4. 준비 (Preparation)
    return "Preparation"


def draw_shorts(canvas, l_hip, r_hip, l_knee, r_knee, head_r):
    """스틱맨에게 골반과 허벅지 상단을 덮는 흰색 반바지를 렌더링."""
    l_hip_arr = np.array(l_hip, dtype=np.float64)
    r_hip_arr = np.array(r_hip, dtype=np.float64)
    l_knee_arr = np.array(l_knee, dtype=np.float64)
    r_knee_arr = np.array(r_knee, dtype=np.float64)
    
    # 반바지는 허벅지 길이의 45%를 덮음
    l_short_end = l_hip_arr + 0.45 * (l_knee_arr - l_hip_arr)
    r_short_end = r_hip_arr + 0.45 * (r_knee_arr - r_hip_arr)
    
    # 반바지 다리 폭은 머리 반경에 비례
    width = head_r * 0.28
    
    # 수직(법선) 벡터 계산 함수
    def get_perp(p1, p2):
        v = p2 - p1
        norm = np.linalg.norm(v) + 1e-6
        u = v / norm
        return np.array([-u[1], u[0]])
        
    l_perp = get_perp(l_hip_arr, l_short_end)
    r_perp = get_perp(r_hip_arr, r_short_end)
    
    # 왼쪽 다리 통 모서리
    l0 = (l_hip_arr - l_perp * width).astype(np.int32)
    l1 = (l_hip_arr + l_perp * width).astype(np.int32)
    l2 = (l_short_end + l_perp * width).astype(np.int32)
    l3 = (l_short_end - l_perp * width).astype(np.int32)
    
    # 오른쪽 다리 통 모서리
    r0 = (r_hip_arr - r_perp * width).astype(np.int32)
    r1 = (r_hip_arr + r_perp * width).astype(np.int32)
    r2 = (r_short_end + r_perp * width).astype(np.int32)
    r3 = (r_short_end - r_perp * width).astype(np.int32)
    
    # 1. 흰색 내부 채우기 (면)
    cv2.fillPoly(canvas, [np.array([l0, l1, l2, l3], dtype=np.int32)], (255, 255, 255), cv2.LINE_AA)
    cv2.fillPoly(canvas, [np.array([r0, r1, r2, r3], dtype=np.int32)], (255, 255, 255), cv2.LINE_AA)
    cv2.fillPoly(canvas, [np.array([l_hip, r_hip, r3, l2], dtype=np.int32)], (255, 255, 255), cv2.LINE_AA)
    
    # 2. 아웃라인 그리기 (선)
    cv2.polylines(canvas, [np.array([l0, l1, l2, l3], dtype=np.int32)], isClosed=True, color=OUTLINE_COLOR, thickness=TORSO_OUTLINE_THICKNESS, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [np.array([r0, r1, r2, r3], dtype=np.int32)], isClosed=True, color=OUTLINE_COLOR, thickness=TORSO_OUTLINE_THICKNESS, lineType=cv2.LINE_AA)
    
    # 허리선
    cv2.line(canvas, l_hip, r_hip, OUTLINE_COLOR, TORSO_OUTLINE_THICKNESS, cv2.LINE_AA)


def draw_phase_badge(canvas, phase_name, w, h):
    """현재 동작 구간(Phase)을 화면 우측 상단에 미려한 네온 테두리 배지로 표시."""
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
    
    # 우측 상단 배치 좌표
    x1 = w - tw - pad * 3
    y1 = pad
    x2 = w - pad
    y2 = y1 + th + pad * 2
    
    # 단계별 고유 네온 컬러 지정
    border_color = (200, 200, 200, 255)
    if "Trophy" in phase_name:
        border_color = (255, 165, 0, 255)    # 주황 (준비완료)
    elif "Drop" in phase_name or "Take" in phase_name:
        border_color = (230, 50, 255, 255)    # 보라 (가속 전 드롭 / 테이크백)
    elif "Impact" in phase_name:
        border_color = (50, 255, 50, 255)     # 연두 (타격 임팩트)
    elif "Follow" in phase_name:
        border_color = (50, 180, 255, 255)    # 하늘 (팔로우 스루)
    elif "Preparation" in phase_name:
        border_color = (180, 180, 180, 255)  # 회색 (동작 대기)
        
    draw.rounded_rectangle([x1, y1, x2, y2], radius=pad, fill=(20, 20, 20, 180),
                           outline=border_color, width=max(1, int(1.5 * LW)))
    draw.text((x1 + pad * 1.5, y1 + pad - tb[1]), text, font=font, fill=(255, 255, 255, 255))
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_stickman(canvas, landmarks, w, h, is_right_handed=True, racket_trail=None, hand_trail=None, phase_smoother=None, is_serve=True):
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

    # 머리·목을 척추(엉덩이→목) 방향으로 배치 → 몸통이 기울면 머리도 따라 기움
    neck_v = np.array(neck, dtype=np.float64)
    spine  = neck_v - np.array(mid_hip, dtype=np.float64)
    sn = np.linalg.norm(spine)
    up = spine / sn if sn > 1e-3 else np.array([0.0, -1.0])
    neck_len = int(head_r * 0.5)
    head_center = neck_v + up * (neck_len + head_r)
    head_cx, head_cy = int(head_center[0]), int(head_center[1])
    head_bottom = (head_center - up * head_r).astype(int)

    # ── 궤적 데이터 추적 및 그리기 ──
    # 라켓 중심점 계산
    racket_wrist = r_wrist if is_right_handed else l_wrist
    racket_elbow = r_elbow if is_right_handed else l_elbow
    wx, wy = racket_wrist
    ex, ey = racket_elbow
    dx, dy = wx - ex, wy - ey
    arm_len = math.sqrt(dx * dx + dy * dy) + 1e-6
    nx, ny = dx / arm_len, dy / arm_len
    grip_length = int(head_r * 1.2)
    frame_ry = int(head_r * 1.5)
    grip_end_x = int(wx + nx * grip_length)
    grip_end_y = int(wy + ny * grip_length)
    head_cx_racket = int(grip_end_x + nx * frame_ry)
    head_cy_racket = int(grip_end_y + ny * frame_ry)
    racket_center = (head_cx_racket, head_cy_racket)

    if racket_trail is not None:
        racket_trail.append(racket_center)
        if len(racket_trail) > 20:
            racket_trail.pop(0)
        draw_glowing_trail(canvas, racket_trail, color=(0, 220, 255)) # 네온 시안
        
    if hand_trail is not None:
        hand_trail.append(racket_wrist)
        if len(hand_trail) > 20:
            hand_trail.pop(0)
        draw_glowing_trail(canvas, hand_trail, color=(255, 150, 50))  # 네온 오렌지

    # 1. 맨 밑바탕: 그림자 그리기
    draw_shadow(canvas, l_ankle, r_ankle)

    # 2. 다리 그리기 (관절 원 포함)
    draw_body_line(canvas, l_hip, l_knee)
    draw_body_line(canvas, l_knee, l_ankle)
    draw_body_line(canvas, r_hip, r_knee)
    draw_body_line(canvas, r_knee, r_ankle)
    
    joint_r = max(LIMB_THICKNESS // 2, int(2 * LW))
    cv2.circle(canvas, l_knee, joint_r, BODY_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, r_knee, joint_r, BODY_COLOR, -1, cv2.LINE_AA)

    # ── 3D Z-depth 기준 동적 신체 부위 레이어 정렬 렌더링 ──
    # 카메라 기준 거리가 더 먼 부위(z가 큰 부위)를 먼저 그리고, 가까운 부위(z가 작은 부위)를 덮어 그립니다.
    # 팔을 상완(Upper)과 전완(Forearm)으로 분리하여 회전 시 가림 현상을 정밀 제어합니다.
    def draw_l_upper_arm():
        draw_body_line(canvas, l_shoulder, l_elbow)
        cv2.circle(canvas, l_elbow, joint_r, BODY_COLOR, -1, cv2.LINE_AA)

    def draw_l_forearm():
        draw_body_line(canvas, l_elbow, l_wrist)
        draw_hand(canvas, l_wrist, hand_r)
        if not is_right_handed:
            draw_racket(canvas, l_wrist, l_elbow, head_r)

    def draw_r_upper_arm():
        draw_body_line(canvas, r_shoulder, r_elbow)
        cv2.circle(canvas, r_elbow, joint_r, BODY_COLOR, -1, cv2.LINE_AA)

    def draw_r_forearm():
        draw_body_line(canvas, r_elbow, r_wrist)
        draw_hand(canvas, r_wrist, hand_r)
        if is_right_handed:
            draw_racket(canvas, r_wrist, r_elbow, head_r)

    def draw_trunk():
        draw_shorts(canvas, l_hip, r_hip, l_knee, r_knee, head_r)
        draw_pentagon_torso(canvas, neck, l_shoulder, r_shoulder, l_hip, r_hip)
        draw_body_line(canvas, neck, tuple(head_bottom), thickness=NECK_THICKNESS)
        draw_head(canvas, head_cx, head_cy, head_r)

    # 미디어파이프 z좌표 추출 및 전면/후면 시점 기반 동적 깊이 보정
    l_upper_z = max(lm[L_SHOULDER].z, lm[L_ELBOW].z)
    l_forearm_z = lm[L_ELBOW].z
    r_upper_z = max(lm[R_SHOULDER].z, lm[R_ELBOW].z)
    r_forearm_z = lm[R_ELBOW].z
    
    # 2D 어깨 위치를 통해 전면/후면 시점 판별 (오른어깨.x > 왼어깨.x 이면 후면 시점)
    is_back_view = lm[R_SHOULDER].x > lm[L_SHOULDER].x
    
    if is_back_view:
        # 후면 시점: 몸통이 팔보다 카메라에 가까움 ➡️ trunk_z를 가장 작게(맨 위에 그림)
        trunk_z = min(l_upper_z, l_forearm_z, r_upper_z, r_forearm_z) - 0.1
    else:
        # 전면 시점: 팔이 몸통 앞에 위치 ➡️ trunk_z를 가장 크게(맨 밑에 그림)
        trunk_z = max(l_upper_z, l_forearm_z, r_upper_z, r_forearm_z) + 0.1

    draw_tasks = [
        (l_upper_z, draw_l_upper_arm),
        (l_forearm_z, draw_l_forearm),
        (r_upper_z, draw_r_upper_arm),
        (r_forearm_z, draw_r_forearm),
        (trunk_z, draw_trunk)
    ]

    # z 기준 내림차순 정렬 (큰 값 = 먼 곳 ➡️ 먼저 그리기)
    draw_tasks.sort(key=lambda x: x[0], reverse=True)

    # 순서대로 그리기 실행
    for depth, draw_func in draw_tasks:
        draw_func()

    # 8. 신발 그리기 (맨 위 레이어)
    draw_shoe(canvas, l_ankle, l_heel, l_foot_idx, l_knee, head_r, is_back_view)
    draw_shoe(canvas, r_ankle, r_heel, r_foot_idx, r_knee, head_r, is_back_view)

    # ── 관절 각도 계산 및 오버레이 그리기 (호 렌더링 및 텍스트 태스크 생성) ──
    draw_angle_tasks = []
    
    # 1. 팔꿈치 각도 (Hitting Arm Elbow)
    # 팔꿈치가 몸통보다 뒤에 있는 경우(z > trunk_z) 투명하게 가려지도록 오버레이를 그리지 않음
    elbow_joint = r_elbow if is_right_handed else l_elbow
    elbow_p1 = r_shoulder if is_right_handed else l_shoulder
    elbow_p2 = r_wrist if is_right_handed else l_wrist
    elbow_z = lm[R_ELBOW].z if is_right_handed else lm[L_ELBOW].z
    
    if elbow_z <= trunk_z:
        task1 = draw_angle_overlay(canvas, elbow_joint, elbow_p1, elbow_p2, color=(0, 165, 255)) # 주황색
        draw_angle_tasks.append(task1)
    
    # 2. 무릎 각도 (Knees)
    task2 = draw_angle_overlay(canvas, l_knee, l_hip, l_ankle, color=(50, 220, 50)) # 연두색
    task3 = draw_angle_overlay(canvas, r_knee, r_hip, r_ankle, color=(50, 220, 50)) # 연두색
    draw_angle_tasks.extend([task2, task3])

    # 3. PIL을 사용하여 깨끗한 아웃라인을 포함한 Unicode 텍스트 일괄 렌더링
    draw_texts_pil(canvas, draw_angle_tasks, font_size=max(int(10 * LW), 10))

    # ── 동작 단계(Phase) 감지 ──
    h_elbow_angle = calculate_angle_2d(elbow_p1, elbow_joint, elbow_p2)
    if is_serve:
        raw_phase = detect_serve_phase(lm, is_right_handed, h_elbow_angle)
    else:
        raw_phase = detect_groundstroke_phase(lm, is_right_handed, h_elbow_angle)
    if phase_smoother is not None:
        current_phase = phase_smoother.add_and_get(raw_phase)
    else:
        current_phase = raw_phase

    return current_phase


def correct_leg_swaps(landmarks, prev_landmarks):
    """MediaPipe가 다리 좌우를 뒤바꿔 인식하는 오류(Swap)를 감지하고 복구합니다."""
    if prev_landmarks is None:
        return landmarks
        
    # 좌우 대응 관절 쌍
    pairs = [
        (23, 24), # L_HIP, R_HIP
        (25, 26), # L_KNEE, R_KNEE
        (27, 28), # L_ANKLE, R_ANKLE
        (29, 30), # L_HEEL, R_HEEL
        (31, 32), # L_FOOT_INDEX, R_FOOT_INDEX
    ]
    
    # 1. 스왑하지 않았을 때의 총 이동 거리 제곱합
    dist_no_swap = 0.0
    for l_idx, r_idx in pairs:
        pl = prev_landmarks[l_idx]
        pr = prev_landmarks[r_idx]
        cl = landmarks[l_idx]
        cr = landmarks[r_idx]
        
        dist_no_swap += (cl.x - pl.x)**2 + (cl.y - pl.y)**2
        dist_no_swap += (cr.x - pr.x)**2 + (cr.y - pr.y)**2
        
    # 2. 좌우를 스왑했을 때의 총 이동 거리 제곱합
    dist_swap = 0.0
    for l_idx, r_idx in pairs:
        pl = prev_landmarks[l_idx]
        pr = prev_landmarks[r_idx]
        cl = landmarks[l_idx]
        cr = landmarks[r_idx]
        
        dist_swap += (cl.x - pr.x)**2 + (cl.y - pr.y)**2
        dist_swap += (cr.x - pl.x)**2 + (cr.y - pl.y)**2
        
    # 스왑했을 때 이동 거리가 현저히 작고, 그 차이가 임계값(관절당 평균 점프) 이상인 경우 스왑 복구
    if dist_swap < dist_no_swap and (dist_no_swap - dist_swap) > 0.015:
        for l_idx, r_idx in pairs:
            lx, ly, lz = landmarks[l_idx].x, landmarks[l_idx].y, landmarks[l_idx].z
            rx, ry, rz = landmarks[r_idx].x, landmarks[r_idx].y, landmarks[r_idx].z
            
            landmarks[l_idx].x, landmarks[l_idx].y, landmarks[l_idx].z = rx, ry, rz
            landmarks[r_idx].x, landmarks[r_idx].y, landmarks[r_idx].z = lx, ly, lz
            
    return landmarks


# ─────────────────────────────────────────
# Step 4: Video Processing
# ─────────────────────────────────────────

def process_video(video_url, name, is_right_handed, label=None, desc=None, speed=1.0):
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

    # 출력 해상도 = OUT_H 기준으로 업스케일 (입력 종횡비 유지, 짝수 폭)
    out_h = OUT_H
    out_w = int(round(orig_w * out_h / orig_h / 2)) * 2
    render_w, render_h = out_w * SSAA, out_h * SSAA
    configure_thickness(render_h)   # 해상도에 맞춘 선 두께

    print(f"[i] Video: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames")
    print(f"[i] Output {out_w}x{out_h}, rendering at {render_w}x{render_h} (SSAA={SSAA}x)")
    if speed != 1.0:
        print(f"[i] Speed Factor: {speed}x (Slow Motion enabled)")
        
    is_serve = "serve" in name.lower()

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
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    smoother  = PoseSmoother(freq=fps if fps > 0 else 30.0)
    frame_idx = 0
    prev_raw_landmarks = None

    # 궤적 및 서브 구간 스무더 초기화
    racket_trail = []
    hand_trail = []
    phase_smoother = PhaseSmoother()

    # 재생 속도 제어용 프레임 보간 카운터
    frame_multiplier = 1.0 / speed
    accumulated_frames = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000.0 / fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        canvas = court_bg.copy()
        current_phase = "Preparation"

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            raw_landmarks = result.pose_landmarks[0]
            mutable_landmarks = [
                SimpleNamespace(x=lm.x, y=lm.y, z=lm.z)
                for lm in raw_landmarks
            ]
            
            # 다리 좌우 스왑 트래킹 오류 복구
            mutable_landmarks = correct_leg_swaps(mutable_landmarks, prev_raw_landmarks)
            prev_raw_landmarks = mutable_landmarks
            
            smoothed = smoother.apply(mutable_landmarks)
            current_phase = draw_stickman(canvas, smoothed, render_w, render_h, is_right_handed,
                                          racket_trail, hand_trail, phase_smoother, is_serve)

        final = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)
        
        # 동작 구간 배지 그리기
        final = draw_phase_badge(final, current_phase, out_w, out_h)
        
        if label:
            final = draw_label(final, label, desc)
            
        # 속도 조절에 따른 프레임 쓰기
        accumulated_frames += frame_multiplier
        write_count = int(accumulated_frames)
        accumulated_frames -= write_count
        for _ in range(write_count):
            out.write(final)

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"  Processing: frame {frame_idx}/{total_frames} ({pct:.1f}%)")

    cap.release()
    out.release()
    landmarker.close()

    print(f"\n[✓] Stickman video saved: {output_path}")
    print(f"    Resolution: {out_w}x{out_h} @ {fps:.1f}fps, {frame_idx} frames")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"  TENNIS STICKMAN ANIMATION GENERATOR v5")
    print(f"  동작: {args.name}  |  손: {'왼손' if args.left else '오른손'}  |  배속: {args.speed}x")
    print("=" * 60)
    process_video(args.url, args.name, is_right_handed=not args.left,
                  label=args.label, desc=args.desc, speed=args.speed)
