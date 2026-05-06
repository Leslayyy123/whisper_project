"""
Live RTSP detector using YAMNet embeddings + trained MLP classifier.
"""

from __future__ import annotations

import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow_hub as hub
import torch
import torch.nn as nn

RTSP_URL = "rtsp://localhost:8554/classroom_audio"
FFMPEG_PATH = (
    r"C:\Users\luvi\Documents\SANDRA\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
)
MODEL_PATH = Path(__file__).resolve().parent / "yamnet_classifier.pt"

INPUT_SAMPLE_RATE = 16000
WINDOW_SECONDS = 1
SUMMARY_SECONDS = 30
DEFAULT_EMBED_DIM = 1024


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden1: int, hidden2: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def read_exactly(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < size:
        chunk = stream.read(size - total)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def extract_acoustic_features(waveform: np.ndarray) -> np.ndarray:
    """Match train_yamnet handcrafted features: fricative_ratio, rms, peak."""
    if waveform.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    spectrum = np.fft.rfft(waveform)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(waveform.size, d=1.0 / INPUT_SAMPLE_RATE)

    total_energy = float(power.sum())
    band_mask = (freqs >= 3000.0) & (freqs <= 8000.0)
    fricative_energy = float(power[band_mask].sum()) if np.any(band_mask) else 0.0
    fricative_ratio = fricative_energy / total_energy if total_energy > 1e-12 else 0.0

    rms = float(np.sqrt(np.mean(waveform * waveform)))
    peak = float(np.max(np.abs(waveform)))
    return np.array([fricative_ratio, rms, peak], dtype=np.float32)


def build_feature_vector(waveform: np.ndarray, yamnet_emb: np.ndarray, embed_dim: int) -> np.ndarray:
    if embed_dim == yamnet_emb.shape[0]:
        return yamnet_emb
    if embed_dim == yamnet_emb.shape[0] + 3:
        return np.concatenate([yamnet_emb, extract_acoustic_features(waveform)], axis=0)
    raise SystemExit(
        f"Unsupported embedding_dim in checkpoint: {embed_dim}. "
        f"Expected {yamnet_emb.shape[0]} or {yamnet_emb.shape[0] + 3}."
    )


def format_label(label: str) -> str:
    if label == "whisper":
        return "[WHISPER DETECTED] ★"
    if label == "room_noise":
        return "[ROOM NOISE]"
    if label == "distractor":
        return "[DISTRACTOR]"
    if label == "normal_speech":
        return "[NORMAL SPEECH]"
    return f"[{label.upper()}]"


def print_summary(history: deque[tuple[str, float]]) -> None:
    counts = {
        "whisper": 0,
        "room_noise": 0,
        "distractor": 0,
        "normal_speech": 0,
    }
    for label, _ in history:
        if label in counts:
            counts[label] += 1
    avg_conf = float(np.mean([conf for _, conf in history])) if history else 0.0

    print("\n=== 30-SECOND SUMMARY ===")
    print(f"Total windows: {len(history)}")
    print(f"Whisper: {counts['whisper']}")
    print(f"Room noise: {counts['room_noise']}")
    print(f"Distractor: {counts['distractor']}")
    print(f"Normal speech: {counts['normal_speech']}")
    print(f"Avg confidence: {avg_conf:.3f}")
    print("=========================\n")


def main() -> None:
    if not Path(FFMPEG_PATH).exists():
        raise SystemExit(f"ffmpeg not found at: {FFMPEG_PATH}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Classifier model not found at: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading classifier checkpoint...")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["classes"]
    hidden1, hidden2 = ckpt["mlp_hidden"]
    embed_dim = int(ckpt.get("embedding_dim", DEFAULT_EMBED_DIM))

    model = MLPClassifier(embed_dim, hidden1, hidden2, len(classes)).to(device)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    print("Loading YAMNet from TensorFlow Hub...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    print("Starting RTSP audio stream...")
    proc = start_ffmpeg_stream()
    if proc.stdout is None:
        stop_process(proc)
        raise SystemExit("Could not open ffmpeg stdout stream.")

    bytes_per_second = INPUT_SAMPLE_RATE * 2 * WINDOW_SECONDS
    history: deque[tuple[str, float]] = deque(maxlen=SUMMARY_SECONDS)
    second_idx = 0

    print("Running live classifier (Ctrl+C to stop)...\n")
    try:
        while True:
            raw = read_exactly(proc.stdout, bytes_per_second)
            if len(raw) < bytes_per_second:
                print("Stream ended or interrupted.")
                break

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            _, embeddings, _ = yamnet_model(samples)
            yamnet_emb = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
            feat = build_feature_vector(samples, yamnet_emb, embed_dim)

            with torch.inference_mode():
                x = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_label = classes[pred_idx]
            conf = float(probs[pred_idx])
            history.append((pred_label, conf))
            second_idx += 1

            print(f"t={second_idx:04d}s {format_label(pred_label)} confidence={conf:.2f}")

            if second_idx % SUMMARY_SECONDS == 0:
                print_summary(history)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        stop_process(proc)


if __name__ == "__main__":
    main()
