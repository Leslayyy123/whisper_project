"""
Record labeled 10-second WAV clips from a live RTSP audio stream.

Requirements:
- ffmpeg available at FFMPEG_PATH
- Python 3.9+
"""

from __future__ import annotations

import math
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

RTSP_URL = "rtsp://localhost:8554/classroom_audio"
FFMPEG_PATH = r"C:\Users\luvi\Documents\SANDRA\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
BASE_DIR = Path(__file__).resolve().parent

CLIP_SECONDS = 10
INPUT_SAMPLE_RATE = 16000


def prompt_label() -> str:
    while True:
        label = input("Enter label (e.g. live_whisper_1m, live_roomnoise): ").strip()
        if label:
            return label
        print("Label cannot be empty.")


def next_file_path(label: str) -> Path:
    out_dir = BASE_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    for wav_path in out_dir.glob(f"{label}_*.wav"):
        stem = wav_path.stem
        if not stem.startswith(f"{label}_"):
            continue
        suffix = stem[len(label) + 1 :]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))

    next_idx = max_idx + 1
    return out_dir / f"{label}_{next_idx:03d}.wav"


def countdown() -> None:
    print("Recording in 3... 2... 1... Recording!")
    time.sleep(1)


def record_clip(output_path: Path) -> None:
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
        str(CLIP_SECONDS),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(INPUT_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def compute_rms(wav_path: Path) -> tuple[float, float]:
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        if n_channels != 1:
            raise ValueError(f"Expected mono WAV, got {n_channels} channels.")
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}.")

        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    if samples.size == 0:
        return 0.0, float("-inf")

    rms = float(np.sqrt(np.mean(samples * samples)))
    rms_norm = rms / 32768.0
    rms_dbfs = 20.0 * math.log10(max(rms_norm, 1e-12))
    return rms_norm, rms_dbfs


def main() -> None:
    if not Path(FFMPEG_PATH).exists():
        raise SystemExit(f"ffmpeg not found at: {FFMPEG_PATH}")

    print("Live RTSP dataset recorder")
    print(f"Stream: {RTSP_URL}")
    print(f"Sample rate: {INPUT_SAMPLE_RATE} Hz, mono, {CLIP_SECONDS}s clips\n")

    label = prompt_label()

    while True:
        output_path = next_file_path(label)
        print(f"\nLabel: {label}")
        print(f"Output: {output_path}")

        countdown()
        try:
            record_clip(output_path)
        except subprocess.CalledProcessError as exc:
            print(f"Recording failed (exit code {exc.returncode}).")
        else:
            print(f"Saved: {output_path}")
            try:
                rms_norm, rms_dbfs = compute_rms(output_path)
                print(f"RMS level: {rms_norm:.6f} ({rms_dbfs:.2f} dBFS)")
            except Exception as err:  # noqa: BLE001
                print(f"Could not compute RMS: {err}")

        action = input(
            "\nPress Enter to record another, type 'switch' to change label, or 'q' to quit: "
        ).strip().lower()

        if action == "q":
            print("Done.")
            break
        if action == "switch":
            label = prompt_label()


if __name__ == "__main__":
    main()
