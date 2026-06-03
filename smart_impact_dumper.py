import cv2
import os
import sys
import json
import math
import argparse
import subprocess
from types import SimpleNamespace

# 미디어파이프 설정용 상수
NOSE = 0
L_SHOULDER = 11; R_SHOULDER = 12
L_ELBOW = 13;    R_ELBOW = 14
L_WRIST = 15;    R_WRIST = 16
L_HIP = 23;      R_HIP = 24
L_KNEE = 25;     R_KNEE = 26
L_ANKLE = 27;    R_ANKLE = 28
L_HEEL = 29;     R_HEEL = 30
L_FOOT_INDEX = 31; R_FOOT_INDEX = 32

def parse_args():
    parser = argparse.ArgumentParser(description="AI 알고리즘 기반 임팩트 후보 프레임 스마트 덤퍼")
    parser.add_argument("video", help="로컬 비디오 파일 경로 (예: C:\\Users\\bagch\\Downloads\\alcaraz_music_bak_input.mp4)")
    parser.add_argument("name", help="동작 이름 (예: alcaraz_music_bak)")
    parser.add_argument("--left", action="store_true", help="왼손잡이 선수 (기본: 오른손잡이)")
    return parser.parse_args()

def calculate_angle_2d(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def detect_serve_phase(lm, is_right_handed, elbow_angle):
    h_wrist = lm[R_WRIST] if is_right_handed else lm[L_WRIST]
    h_shoulder = lm[R_SHOULDER] if is_right_handed else lm[L_SHOULDER]
    h_elbow = lm[R_ELBOW] if is_right_handed else lm[L_ELBOW]
    nh_wrist = lm[L_WRIST] if is_right_handed else lm[R_WRIST]
    nh_shoulder = lm[L_SHOULDER] if is_right_handed else lm[R_SHOULDER]
    nose = lm[NOSE]
    
    if h_wrist.y < nose.y and elbow_angle > 155:
        return "Impact"
    return "Preparation"

# 오디오 타구음 피크 검출 (3번 알고리즘 중 오디오 영역)
def detect_impacts_from_audio(video_path, fps, n_frames, min_gap_sec=1.0, threshold_ratio=0.20):
    import tempfile, wave
    import numpy as np
    print("[i] 오디오 타구음 분석 중...")
    tmp_wav = tempfile.mktemp(suffix='.wav')
    try:
        # ffmpeg이 시스템 PATH 혹은 다운로드 폴더에 있으므로 호출 시도
        # 다운로드 폴더에 있는 ffmpeg.exe를 직접 사용할 수 있도록 안전장치 추가
        ffmpeg_exe = "ffmpeg"
        if os.path.exists(r"C:\Users\bagch\Downloads\ffmpeg.exe"):
            ffmpeg_exe = r"C:\Users\bagch\Downloads\ffmpeg.exe"
            
        subprocess.run([
            ffmpeg_exe, '-y', '-i', video_path,
            '-ac', '1', '-ar', '44100',
            '-c:a', 'pcm_s16le', tmp_wav
        ], capture_output=True, check=True)
    except Exception as e:
        print(f"[!] 오디오 추출 실패: {e}")
        return []

    with wave.open(tmp_wav, 'rb') as wf:
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    win = int(sample_rate * 0.020)
    hop = int(sample_rate * 0.005)
    n_wins = (len(samples) - win) // hop
    energy = np.array([
        np.sum(samples[i*hop:i*hop+win] ** 2) / win
        for i in range(n_wins)
    ])

    onset = np.diff(energy)
    onset = np.clip(onset, 0, None)
    if onset.max() > 0:
        onset /= onset.max()

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

    video_frames = []
    for pk in peaks:
        t = pk * hop / sample_rate
        f = min(int(t * fps), n_frames - 1)
        video_frames.append(f)
    return video_frames

def main():
    args = parse_args()
    video_path = args.video
    name = args.name
    is_right_handed = not args.left

    if not os.path.exists(video_path):
        print(f"[ERROR] 비디오 파일을 찾을 수 없습니다: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 캐시 파일 위치 파악
    # text_to_video_project 내부 또는 Downloads 폴더 중 캐시가 있는지 탐색
    cache_candidates = [
        f"{name}_pose_cache.json",
        os.path.join(r"c:\Users\bagch\Downloads\text_to_video_project", f"{name}_pose_cache.json"),
        os.path.join(r"C:\Users\bagch\Downloads", f"{name}_pose_cache.json")
    ]
    
    cache_path = None
    for cp in cache_candidates:
        if os.path.exists(cp):
            cache_path = cp
            break

    if not cache_path:
        print("[ERROR] 포즈 캐시 파일을 찾을 수 없습니다. tennis_stickman_v10.py를 먼저 1회 실행하여 포즈 캐시를 만드십시오.")
        return

    print(f"[OK] Loading pose landmarks from cache: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        cached_data = json.load(f)

    all_smoothed_landmarks = []
    for frame_data in cached_data:
        if frame_data is None:
            all_smoothed_landmarks.append(None)
        else:
            all_smoothed_landmarks.append([
                SimpleNamespace(x=lm["x"], y=lm["y"], z=lm["z"], v=lm.get("v", 1.0))
                for lm in frame_data
            ])

    global np
    import numpy as np

    candidates = set()
    is_serve = "serve" in name.lower()

    # ─────────────────────────────────────────
    # 알고리즘 3: 오디오 + 포즈 이중 검증
    # ─────────────────────────────────────────
    audio_peaks = detect_impacts_from_audio(video_path, fps, total_frames)
    if audio_peaks:
        h_shoulder_idx = R_SHOULDER if is_right_handed else L_SHOULDER
        h_wrist_idx    = R_WRIST    if is_right_handed else L_WRIST
        h_elbow_idx    = R_ELBOW    if is_right_handed else L_ELBOW

        for af in audio_peaks:
            best_f = None
            best_score = -1
            # 오디오 피크 전후 10프레임 범위에서 생체역학 점수 채점
            for fi in range(max(0, af - 10), min(total_frames, af + 10)):
                sm = all_smoothed_landmarks[fi]
                if sm is None:
                    continue
                s = sm[h_shoulder_idx]
                w = sm[h_wrist_idx]
                e = sm[h_elbow_idx]

                # 어깨 높이 부근 타구 범위 검사
                wrist_away_from_shoulder = abs(w.x - s.x) > 0.03
                wrist_in_height_zone     = (s.y - 0.05 <= w.y <= s.y + 0.30)
                if not (wrist_away_from_shoulder and wrist_in_height_zone):
                    continue

                elbow_angle = calculate_angle_2d((s.x, s.y), (e.x, e.y), (w.x, w.y))
                if elbow_angle < 120.0:
                    continue

                # 채점
                arm_extension = math.sqrt((w.x - s.x)**2 + (w.y - s.y)**2)
                x_offset = abs(w.x - s.x)
                x_score   = max(0.0, 1.0 - abs(x_offset - 0.135) / 0.135)
                ext_score = max(0.0, 1.0 - abs(arm_extension - 0.140) / 0.100)
                ang_score = max(0.0, 1.0 - abs(elbow_angle - 156.0) / 30.0)
                score = x_offset + (x_score + ext_score + ang_score) * 0.1

                if score > best_score:
                    best_score = score
                    best_f = fi
            if best_f is not None:
                candidates.add(best_f)
                print(f"[AI 후보군] 3번 오디오+포즈 이중 검증 검출: Frame {best_f} (소리피크={af}, 점수={best_score:.3f})")

    # ─────────────────────────────────────────
    # 알고리즘 4: 팔 신장도 최대 감지 (지상/서브)
    # ─────────────────────────────────────────
    if not is_serve:
        # 포/백핸드 신장도 감지
        h_shoulder_idx = R_SHOULDER if is_right_handed else L_SHOULDER
        h_wrist_idx    = R_WRIST    if is_right_handed else L_WRIST
        
        extensions = []
        for fi in range(total_frames):
            sm = all_smoothed_landmarks[fi]
            if sm is None:
                extensions.append(0.0)
                continue
            s = sm[h_shoulder_idx]
            w = sm[h_wrist_idx]
            extensions.append(math.sqrt((w.x - s.x)**2 + (w.y - s.y)**2))

        # 속도 계산 (스윙 활성화 확인용)
        win = 3
        abs_velocities = []
        for fi in range(total_frames):
            a, b = fi - win, fi + win
            if a < 0 or b >= total_frames or all_smoothed_landmarks[a] is None or all_smoothed_landmarks[b] is None:
                abs_velocities.append(0.0)
                continue
            wa = all_smoothed_landmarks[a][h_wrist_idx]
            wb = all_smoothed_landmarks[b][h_wrist_idx]
            abs_velocities.append(math.sqrt((wb.x - wa.x)**2 + (wb.y - wa.y)**2) / (2 * win))

        max_abs = max(abs_velocities) if abs_velocities else 0
        vel_threshold = max_abs * 0.12

        in_swing = False
        swing_start = 0
        swing_windows = []
        for fi in range(total_frames):
            if not in_swing and abs_velocities[fi] >= vel_threshold:
                in_swing, swing_start = True, fi
            elif in_swing and abs_velocities[fi] < vel_threshold * 0.25:
                if fi - swing_start >= 6:
                    swing_windows.append((swing_start, fi))
                in_swing = False
        if in_swing:
            swing_windows.append((swing_start, total_frames - 1))

        # 인접 구간 병합
        merged = []
        for ws, we in swing_windows:
            if merged and ws - merged[-1][1] <= 20:
                merged[-1] = (merged[-1][0], we)
            else:
                merged.append((ws, we))

        # 신장도 50% 지점 검출
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
                wrist_y = sm[h_wrist_idx].y
                shoulder_y = sm[h_shoulder_idx].y
                if wrist_y < shoulder_y - 0.04:
                    continue # 어깨보다 손이 높은 오탐 방지
            
            candidates.add(impact_f)
            print(f"[AI 후보군] 4번 팔 신장도 분석 검출: Frame {impact_f}")

    else:
        # 서브 분석 검출
        serve_candidates = []
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
                    serve_candidates.append(frame_idx)
        if serve_candidates:
            groups = []
            current_group = [serve_candidates[0]]
            for val in serve_candidates[1:]:
                if val == current_group[-1] + 1:
                    current_group.append(val)
                else:
                    groups.append(current_group)
                    current_group = [val]
            groups.append(current_group)
            for g in groups:
                mid_f = g[len(g) // 2]
                shifted_mid_f = max(0, mid_f - 3)
                candidates.add(shifted_mid_f)
                print(f"[AI 후보군] 4번 서브 모션 분석 검출: Frame {shifted_mid_f}")

    # 초반 오검출 (1.5초 이내) 제거
    early_cutoff = int(round(fps * 1.5))
    final_candidates = sorted([c for c in candidates if c > early_cutoff])

    if not final_candidates:
        print("[!] 검출된 임팩트 후보 프레임이 없습니다.")
        return

    print("=" * 60)
    print(f"[결과] 총 {len(final_candidates)}개의 고유 임팩트 후보군 추출 완료: {final_candidates}")
    print("=" * 60)

    # ─────────────────────────────────────────
    # 캡처 및 이미지 덤프 (전후 3프레임 덤프)
    # ─────────────────────────────────────────
    output_dir = os.path.join(r"C:\Users\bagch\Downloads", f"{name}_candidates")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[i] 후보 프레임 캡처 시작 -> 저장 폴더: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    
    # 덤프할 프레임 번호들과 그 소속 후보군 매핑
    # 예: 후보 320번 -> 317~323번 프레임 추출 필요
    frames_to_dump = {}
    for c in final_candidates:
        for offset in range(-3, 4):
            target_f = c + offset
            if 0 <= target_f < total_frames:
                if target_f not in frames_to_dump:
                    frames_to_dump[target_f] = []
                frames_to_dump[target_f].append((c, offset))

    frame_idx = 0
    dumped_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in frames_to_dump:
            # 해당 프레임을 저장
            for candidate_f, offset in frames_to_dump[frame_idx]:
                sign = "+" if offset >= 0 else ""
                out_name = f"candidate_f{candidate_f}_offset_{sign}{offset}_(actual_f{frame_idx}).png"
                out_path = os.path.join(output_dir, out_name)
                
                # 가독성을 위해 이미지 좌측 상단에 텍스트 워터마크 추가
                annotated = frame.copy()
                cv2.putText(annotated, f"Candidate: {candidate_f} | Frame: {frame_idx} (Offset: {sign}{offset})", 
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 4, cv2.LINE_AA)
                cv2.putText(annotated, f"Candidate: {candidate_f} | Frame: {frame_idx} (Offset: {sign}{offset})", 
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imwrite(out_path, annotated)
                dumped_count += 1
                
        frame_idx += 1
        if frame_idx > max(frames_to_dump.keys()):
            break

    cap.release()
    print(f"[OK] 성공적으로 {dumped_count}장의 후보 이미지 캡처를 저장했습니다!")
    print(f"\n[사용 안내]")
    print(f"1. Downloads/{name}_candidates 폴더 내 이미지들을 열어서 가장 공이 라켓에 닿아 있는 실제 프레임 번호(actual_f값)를 고르세요.")
    print(f"2. 그 번호 N을 활용하여 아래와 같이 최종 비디오를 렌더링하면 완벽합니다:")
    print(f"   python tennis_stickman_v10.py {name}_input.mp4 {name} --strobe --strobe-frames 20 --strobe-step 2 --impact-frame N")
    print("=" * 60)

if __name__ == "__main__":
    main()
