"""
Live whisper detector: RTSP audio → wav2vec2-base features → trained MLP (whisper_classifier.pt).

Replaces a heuristic whisper_detector.py: all decisions come from the saved classifier, not
hand-tuned energy/voice thresholds.

Requires: FFmpeg executable below (or edit FFMPEG_EXE). PyTorch+CUDA recommended.
"""

from __future__ import annotations

import collections
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

CHECKPOINT_PATH = Path(__file__).resolve().parent / "whisper_classifier.pt"

RTSP_URL = "rtsp://localhost:8554/classroom_audio"
INPUT_SAMPLE_RATE = 16000
FFMPEG_EXE = r"C:\Users\Leslie\Thesis\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

SUMMARY_INTERVAL_SEC = 30
DROPOUT = 0.1

# class name -> (print label, is_whisper_alert)
PRINT_LABELS: dict[str, tuple[str, bool]] = {
    "whisper": ("WHISPER DETECTED", True),
    "room_noise": ("ROOM NOISE", False),
    "distractor": ("DISTRACTOR", False),
    "normal_speech": ("NORMAL SPEECH", False),
}


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


class MLPClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h1: int,
        h2: int,
        n_classes: int,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_models(device: torch.device) -> tuple[
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    MLPClassifier,
    list[str],
]:
    if not CHECKPOINT_PATH.is_file():
        raise SystemExit(f"Missing checkpoint: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    classes: list[str] = ckpt["classes"]
    h1, h2 = ckpt["mlp_hidden"]
    feature_dim = ckpt["feature_dim"]
    model_id = ckpt.get("wav2vec_model_id", "facebook/wav2vec2-base")

    feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    w2v = Wav2Vec2Model.from_pretrained(model_id).to(device)
    for p in w2v.parameters():
        p.requires_grad = False
    w2v.eval()

    mlp = MLPClassifier(feature_dim, h1, h2, len(classes)).to(device)
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    mlp.eval()

    return feat_extractor, w2v, mlp, classes


@torch.inference_mode()
def wav2vec_pooled_feature(
    samples_float: np.ndarray,
    device: torch.device,
    feat_extractor: Wav2Vec2FeatureExtractor,
    w2v: Wav2Vec2Model,
    sampling_rate: int,
) -> torch.Tensor:
    """Mean-pooled last hidden state for one waveform; returns [768] on device."""
    inputs = feat_extractor(
        [samples_float.astype(np.float32)],
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = w2v(**inputs).last_hidden_state
    mask = inputs.get("attention_mask")
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
    else:
        pooled = out.mean(dim=1)
    return pooled.squeeze(0)


def build_ffmpeg_command(rtsp_url: str) -> list[str]:
    return [
        FFMPEG_EXE,
        "-nostdin",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(INPUT_SAMPLE_RATE),
        "-",
    ]


def pcm_s16le_bytes_to_float_mono(raw: bytes) -> np.ndarray:
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return np.ascontiguousarray(x)


def print_summary(
    classes: list[str],
    preds_in_period: list[int],
) -> None:
    whisper_i = classes.index("whisper")
    whisper_secs = sum(1 for p in preds_in_period if p == whisper_i)
    non = [p for p in preds_in_period if p != whisper_i]
    if non:
        counts = collections.Counter(non)
        most_i, _ = counts.most_common(1)[0]
        most_name = classes[most_i]
    else:
        most_name = "(none)"

    print(
        f"\n--- 30s summary: whisper_seconds={whisper_secs} / {len(preds_in_period)}  "
        f"most_common_non_whisper={most_name} ---\n",
        flush=True,
    )


def main() -> None:
    ensure_utf8_stdio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available; using CPU.", file=sys.stderr)

    if not Path(FFMPEG_EXE).is_file():
        raise SystemExit(f"ffmpeg not found: {FFMPEG_EXE}")

    print(f"Loading models on {device}...", flush=True)
    feat_extractor, w2v, mlp, classes = load_models(device)
    target_sr = feat_extractor.sampling_rate
    if target_sr != INPUT_SAMPLE_RATE:
        print(
            f"Warning: feature extractor expects sr={target_sr}, "
            f"INPUT_SAMPLE_RATE={INPUT_SAMPLE_RATE}.",
            file=sys.stderr,
        )

    cmd = build_ffmpeg_command(RTSP_URL)
    print(f"Starting ffmpeg → {RTSP_URL}", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    assert proc.stdout is not None

    bytes_per_sec = INPUT_SAMPLE_RATE * 2
    buf = bytearray()
    period_preds: list[int] = []
    period_start = time.monotonic()

    try:
        while True:
            chunk = proc.stdout.read(8192)
            if chunk is None:
                break
            if len(chunk) == 0:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                raise SystemExit(f"ffmpeg ended (exit {proc.wait()}). stderr: {err!r}")

            buf.extend(chunk)
            while len(buf) >= bytes_per_sec:
                frame = bytes(buf[:bytes_per_sec])
                del buf[:bytes_per_sec]

                audio = pcm_s16le_bytes_to_float_mono(frame)
                if audio.shape[0] != INPUT_SAMPLE_RATE:
                    continue

                with torch.inference_mode():
                    feat = wav2vec_pooled_feature(
                        audio, device, feat_extractor, w2v, INPUT_SAMPLE_RATE
                    ).unsqueeze(0)
                    logits = mlp(feat)
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                pred_i = int(torch.argmax(probs).item())
                conf = float(probs[pred_i].item())
                class_name = classes[pred_i]

                period_preds.append(pred_i)
                now = time.monotonic()
                if now - period_start >= SUMMARY_INTERVAL_SEC:
                    print_summary(classes, period_preds)
                    period_preds.clear()
                    period_start = now

                tag, is_alert = PRINT_LABELS[class_name]
                if is_alert:
                    print(f"[{tag}] confidence={conf:.2f} ★", flush=True)
                else:
                    print(f"[{tag}] confidence={conf:.2f}", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
