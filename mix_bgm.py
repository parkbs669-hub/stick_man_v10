import sys
import os
import argparse

# Try importing moviepy components and install if missing
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
except ImportError:
    try:
        from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
    except ImportError:
        print("[i] Installing moviepy...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "moviepy"], check=True)
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
        except ImportError:
            from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

def mix_volume(audio_clip, factor):
    if hasattr(audio_clip, 'volumex'):
        return audio_clip.volumex(factor)
    elif hasattr(audio_clip, 'multiply_volume'):
        return audio_clip.multiply_volume(factor)
    elif hasattr(audio_clip, 'fx'):
        try:
            from moviepy.audio.fx.all import volumex
            return audio_clip.fx(volumex, factor)
        except Exception:
            pass
    return audio_clip

def subclip_audio(audio_clip, end_time):
    end_time = min(end_time, audio_clip.duration)
    if hasattr(audio_clip, 'subclipped'):
        return audio_clip.subclipped(0, end_time)
    else:
        return audio_clip.subclip(0, end_time)

def set_audio_safe(video_clip, audio_clip):
    if hasattr(video_clip, 'with_audio'):
        return video_clip.with_audio(audio_clip)
    elif hasattr(video_clip, 'set_audio'):
        return video_clip.set_audio(audio_clip)
    return video_clip

def set_start_safe(clip, start_time):
    if hasattr(clip, 'with_start'):
        return clip.with_start(start_time)
    elif hasattr(clip, 'set_start'):
        return clip.set_start(start_time)
    return clip

def parse_args():
    parser = argparse.ArgumentParser(description="Mix tennis stickman animation, court sound, and BGM with impact SFX.")
    parser.add_argument("name", nargs="?", default="tennis_shot_3", help="Base name of the video (default: tennis_shot_3)")
    parser.add_argument("--impact-frames", type=int, nargs="+", default=None, metavar="N", help="Impact frame numbers to inject hit SFX (e.g., --impact-frames 38 218)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate for calculating timestamps (default: 30.0)")
    return parser.parse_args()

def generate_tennis_hit_sound(out_path):
    """파이썬 내장 라이브러리만을 활용해 매우 과격하고 폭발적인 타구 효과음(WAV) 합성"""
    import wave, struct, math, random
    sample_rate = 44100
    duration = 0.40  # 폭발적인 잔향을 위해 400ms로 연장
    n_samples = int(sample_rate * duration)
    
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        
        # 1. 과격하고 날카로운 채찍 타격음 (Shockwave Chirp - 20ms)
        t_chirp = 0.020
        if t < t_chirp:
            f_start = 3500.0
            f_end = 150.0
            k = math.log(f_end / f_start) / t_chirp
            phase = 2 * math.pi * f_start * (math.exp(k * t) - 1) / k
            crack = math.sin(phase) * math.exp(-120 * t) * 1.5
        else:
            crack = 0.0

        # 2. 강력한 폭발성 소음 (Explosive Noise Blast)
        noise = random.uniform(-1.0, 1.0) * math.exp(-80 * t) * 1.2
        
        # 3. 묵직하고 거대한 베이스 붐 (Sub-bass Cinematic Boom - 300ms)
        t_boom = 0.30
        if t < t_boom:
            k_boom = math.log(30.0 / 180.0) / t_boom
            phase_boom = 2 * math.pi * 180.0 * (math.exp(k_boom * t) - 1) / k_boom
            boom = math.sin(phase_boom) * math.exp(-12 * t) * 1.2
        else:
            boom = 0.0
            
        # 4. 스트링의 강한 금속성 울림 (Metallic Ping)
        string_fundamental = math.sin(2 * math.pi * 500 * t) * math.exp(-25 * t) * 0.5
        string_harmonic = math.sin(2 * math.pi * 1000 * t) * math.exp(-40 * t) * 0.3
        
        # 성분 결합 (최대 진폭이 매우 높음)
        combined = crack + noise + boom + string_fundamental + string_harmonic
        
        # 5. 과격한 디스토션 및 오버드라이브 적용 (Soft-Clipping Saturation)
        val = math.tanh(combined * 2.2)
        
        int_val = int(val * 32767)
        samples.append(int_val)
        
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for val in samples:
            w.writeframesraw(struct.pack("<h", val))
    print(f"[INFO] Synthesized explosive tennis hit SFX saved at: {out_path}")

def main():
    args = parse_args()
    name = args.name
    
    video_path = rf"C:\Users\bagch\Downloads\{name}_stickman.mp4"
    orig_video_path = rf"C:\Users\bagch\Downloads\{name}_input.mp4"
    bgm_path = r"C:\Users\bagch\Downloads\bgm.mp3"
    output_path = rf"C:\Users\bagch\Downloads\{name}_stickman_with_audio.mp4"
    sfx_path = r"C:\Users\bagch\Downloads\tennis_hit_sfx.wav"

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return
    if not os.path.exists(orig_video_path):
        print(f"[ERROR] Original video file not found: {orig_video_path}")
        return

    print("[INFO] Loading video clip...")
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration

    print("[INFO] Loading original video audio (court sounds)...")
    orig_video = VideoFileClip(orig_video_path)
    court_audio = orig_video.audio

    audio_sources = []
    
    # 원본 비디오 사운드에 저작권 음악(BTS 등)이 기본으로 깔려 있으므로 완전히 음소거합니다.
    print("[INFO] Muting original video audio to prevent copyright issues (BTS etc.).")

    # Check if BGM exists and mix it at 100% volume
    if os.path.exists(bgm_path):
        print("[INFO] bgm.mp3 found. Adding BGM at 100% volume as the sole audio track...")
        bgm_audio = AudioFileClip(bgm_path)
        bgm_audio_sub = subclip_audio(bgm_audio, video_duration)
        # 배경 음악 볼륨을 대폭 낮춰 타구음이 압도적으로 크게 들리도록 조정 (0.75 -> 0.3)
        bgm_audio_scaled = mix_volume(bgm_audio_sub, 0.3)
        audio_sources.append(bgm_audio_scaled)
    else:
        print("[INFO] bgm.mp3 not found. Video will be mixed without background music.")

    # 임팩트 효과음 합성 및 배치
    if args.impact_frames:
        if not os.path.exists(sfx_path):
            generate_tennis_hit_sound(sfx_path)
            
        print(f"[INFO] Injecting hit SFX at frames: {args.impact_frames}")
        for iframe in args.impact_frames:
            # 프레임 인덱스 -> 시간(초) 환산
            hit_time = iframe / args.fps
            if hit_time <= video_duration:
                hit_clip = set_start_safe(AudioFileClip(sfx_path), hit_time)
                # 타구 효과음 볼륨을 8배로 극대화 증폭
                hit_clip_scaled = mix_volume(hit_clip, 8.0)
                audio_sources.append(hit_clip_scaled)

    if audio_sources:
        print("[INFO] Mixing audio channels...")
        final_audio = CompositeAudioClip(audio_sources)
        video_with_audio = set_audio_safe(video_clip, final_audio)
        
        print(f"[INFO] Writing output video to: {output_path} ...")
        # Set logger=None for moviepy to prevent terminal noise
        try:
            video_with_audio.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                logger=None
            )
        except TypeError:
            video_with_audio.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac"
            )
            
        final_audio.close()
        video_with_audio.close()
    else:
        print("[WARN] No audio sources to mix. Copying video without audio.")
        video_clip.write_videofile(output_path, codec="libx264")

    # Clean up clips
    video_clip.close()
    orig_video.close()
    if os.path.exists(bgm_path) and 'bgm_audio' in locals():
        bgm_audio.close()

    print(f"[SUCCESS] Audio mixing complete! Output saved at: {output_path}")

if __name__ == "__main__":
    main()
