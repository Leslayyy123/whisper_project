"""
Live RTSP whisper detector using YAMNet.

Requirements:
- tensorflow
- tensorflow_hub
- ffmpeg at FFMPEG_PATH
"""

from __future__ import annotations

import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

RTSP_URL = "rtsp://localhost:8554/classroom_audio"
FFMPEG_PATH = (
    r"C:\Users\luvi\Documents\SANDRA\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
)
INPUT_SAMPLE_RATE = 16000
WINDOW_SECONDS = 1
SUMMARY_SECONDS = 30
RETRY_WAIT_SECONDS = 2
MAX_RETRIES = 3

WHISPER_TERMS = ("whisper", "whispering")
SPEECH_TERMS = ("speech", "talk", "talking", "conversation", "speaking")


def load_yamnet() -> tuple[hub.KerasLayer, list[str]]:
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names: list[str] = []
    with tf.io.gfile.GFile(class_map_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3:
                class_names.append(parts[2].strip().strip('"'))
    return model, class_names


def start_ffmpeg_stream() -> subprocess.Popen[bytes]:
    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        RTSP_URL,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(INPUT_SAMPLE_RATE),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def stop_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def check_rtsp_connection() -> bool:
    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        RTSP_URL,
        "-t",
        "1",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(INPUT_SAMPLE_RATE),
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def read_exactly(stream, size: int) -> bytes:
    """Read exactly size bytes unless stream closes."""
    chunks: list[bytes] = []
    total = 0
    while total < size:
        chunk = stream.read(size - total)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def classify_window(
    waveform: np.ndarray, model: hub.KerasLayer, class_names: list[str]
) -> tuple[str, float, list[tuple[str, float]]]:
    scores, _, _ = model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_indices = np.argsort(mean_scores)[-3:][::-1]
    top3 = [(class_names[i], float(mean_scores[i])) for i in top_indices]

    whisper_score = 0.0
    speech_score = 0.0
    for name, score in top3:
        lname = name.lower()
        if any(term in lname for term in WHISPER_TERMS):
            whisper_score = max(whisper_score, score)
        if any(term in lname for term in SPEECH_TERMS):
            speech_score = max(speech_score, score)

    if whisper_score > 0:
        return "[WHISPER DETECTED] ★", whisper_score, top3
    if speech_score > 0:
        return "[SPEECH DETECTED]", speech_score, top3
    return "[BACKGROUND]", top3[0][1], top3


def print_summary(history: deque[tuple[str, float]]) -> None:
    whisper_count = sum(1 for label, _ in history if label == "[WHISPER DETECTED] ★")
    speech_count = sum(1 for label, _ in history if label == "[SPEECH DETECTED]")
    background_count = sum(1 for label, _ in history if label == "[BACKGROUND]")
    avg_conf = float(np.mean([conf for _, conf in history])) if history else 0.0
    print("\n=== 30-SECOND SUMMARY ===")
    print(f"Total windows: {len(history)}")
    print(f"Whisper: {whisper_count}")
    print(f"Speech: {speech_count}")
    print(f"Background: {background_count}")
    print(f"Avg confidence: {avg_conf:.3f}")
    print("=========================\n")


def main() -> None:
    if not Path(FFMPEG_PATH).exists():
        raise SystemExit(f"ffmpeg not found at: {FFMPEG_PATH}")

    print("Checking RTSP stream connection...")
    if check_rtsp_connection():
        print("RTSP connection OK.")
    else:
        print("RTSP connection failed at startup.")
        print("Make sure mediamtx and audio_pipeline.py are running!")
        raise SystemExit("Could not connect to RTSP stream.")

    print("Loading YAMNet from TensorFlow Hub...")
    model, class_names = load_yamnet()

    print("Starting RTSP audio stream...")
    proc = start_ffmpeg_stream()
    if proc.stdout is None:
        stop_process(proc)
        raise SystemExit("Could not open ffmpeg stdout stream.")

    bytes_per_second = INPUT_SAMPLE_RATE * 2 * WINDOW_SECONDS
    history: deque[tuple[str, float]] = deque(maxlen=SUMMARY_SECONDS)
    second_idx = 0
    retries = 0

    print("Running detection (Ctrl+C to stop)...\n")
    try:
        while True:
            raw = read_exactly(proc.stdout, bytes_per_second)
            if len(raw) < bytes_per_second:
                retries += 1
                print(
                    f"Stream ended unexpectedly. Retrying in {RETRY_WAIT_SECONDS}s "
                    f"({retries}/{MAX_RETRIES})..."
                )
                stop_process(proc)
                if retries >= MAX_RETRIES:
                    print("Max retries reached. Giving up.")
                    print("Make sure mediamtx and audio_pipeline.py are running!")
                    break
                time.sleep(RETRY_WAIT_SECONDS)
                proc = start_ffmpeg_stream()
                if proc.stdout is None:
                    stop_process(proc)
                    print("Failed to reopen RTSP stream output.")
                    print("Make sure mediamtx and audio_pipeline.py are running!")
                    break
                continue

            retries = 0

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            label, confidence, top3 = classify_window(samples, model, class_names)
            second_idx += 1
            history.append((label, confidence))

            top3_str = ", ".join([f"{name} ({score:.3f})" for name, score in top3])
            print(f"t={second_idx:04d}s {label} conf={confidence:.3f} | top3: {top3_str}")

            if second_idx % SUMMARY_SECONDS == 0:
                print_summary(history)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        stop_process(proc)


if __name__ == "__main__":
    main()
