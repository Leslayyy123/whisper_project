"""
Evaluate whisper_classifier.pt on the same 20% validation split as training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Must match train_classifier.py stratified split
SEED = 42
VAL_FRACTION = 0.2

METADATA_PATH = Path(__file__).resolve().parent / "metadata.csv"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "whisper_classifier.pt"

FEATURE_BATCH = 8
DROPOUT = 0.1

CLASS_NAMES = ["whisper", "room_noise", "distractor", "normal_speech"]


def ensure_utf8_stdio() -> None:
    """Avoid UnicodeEncodeError on Windows (cp1252) when printing ✅/❌."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def load_waveform(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


class MLPClassifier(nn.Module):
    """Same layout as training; hidden sizes from checkpoint."""

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


@torch.inference_mode()
def extract_wav2vec_features(
    paths: list[str],
    device: torch.device,
    feat_extractor: Wav2Vec2FeatureExtractor,
    w2v: Wav2Vec2Model,
    target_sr: int,
) -> np.ndarray:
    w2v.eval()
    out_list: list[np.ndarray] = []
    for start in range(0, len(paths), FEATURE_BATCH):
        batch_paths = paths[start : start + FEATURE_BATCH]
        waves = [load_waveform(p, target_sr) for p in batch_paths]
        inputs = feat_extractor(
            [w.numpy() for w in waves],
            sampling_rate=target_sr,
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
        out_list.append(pooled.cpu().numpy())
    return np.concatenate(out_list, axis=0)


def main() -> None:
    ensure_utf8_stdio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available; using CPU.", file=sys.stderr)

    if not CHECKPOINT_PATH.is_file():
        raise SystemExit(f"Missing checkpoint: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    classes: list[str] = ckpt["classes"]
    h1, h2 = ckpt["mlp_hidden"]
    feature_dim = ckpt["feature_dim"]
    model_id = ckpt.get("wav2vec_model_id", "facebook/wav2vec2-base")

    idx_to_name = {i: c for i, c in enumerate(classes)}
    model = MLPClassifier(feature_dim, h1, h2, len(classes)).to(device)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    df = pd.read_csv(METADATA_PATH)
    if "file_path" not in df.columns or "label" not in df.columns:
        raise SystemExit("metadata.csv must have columns: file_path, label")

    bad = set(df["label"].unique()) - set(classes)
    if bad:
        raise SystemExit(f"Unknown labels in CSV: {bad}")

    paths = df["file_path"].astype(str).tolist()
    y = df["label"].map(LABEL_TO_IDX).astype(np.int64).values

    # Same validation indices as train_classifier.py (stratified 80/20, seed 42)
    idx_all = np.arange(len(df))
    _, val_idx = train_test_split(
        idx_all,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=y,
    )
    val_paths = [paths[i] for i in val_idx]
    y_val = y[val_idx]

    print(f"Loading {model_id} feature extractor on {device}...")
    feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    w2v = Wav2Vec2Model.from_pretrained(model_id).to(device)
    for p in w2v.parameters():
        p.requires_grad = False
    target_sr = feat_extractor.sampling_rate

    print(f"Extracting features for {len(val_paths)} validation files (sr={target_sr})...")
    X_val = extract_wav2vec_features(val_paths, device, feat_extractor, w2v, target_sr)
    del w2v
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.inference_mode():
        logits = model(torch.from_numpy(X_val).float().to(device))
    pred = logits.argmax(dim=1).cpu().numpy()

    overall_acc = accuracy_score(y_val, pred)
    cm = confusion_matrix(y_val, pred, labels=list(range(len(classes))))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    print(f"\nValidation set: {len(val_paths)} files (same 20% split as training, seed={SEED})\n")
    print("Per file (argmax prediction):")
    print("-" * 100)

    rows = []
    for path_str, yt, yp in zip(val_paths, y_val, pred):
        name = Path(path_str).name
        true_name = idx_to_name[int(yt)]
        pred_name = idx_to_name[int(yp)]
        ok = yt == yp
        rows.append((name, true_name, pred_name, ok))

    for name, true_name, pred_name, ok in sorted(rows, key=lambda r: r[0].lower()):
        mark = "✅" if ok else "❌"
        print(f"{name:<45}  true={true_name:<15}  pred={pred_name:<15}  {mark}")

    print("-" * 100)
    print("\nPer-class accuracy (validation):")
    for c_idx, cname in enumerate(classes):
        mask = y_val == c_idx
        n = int(mask.sum())
        if n == 0:
            print(f"  {cname}: 0/0 (no samples)")
            continue
        correct = int((pred[mask] == y_val[mask]).sum())
        print(f"  {cname}: {correct}/{n} correct")

    print(f"\nOverall accuracy: {overall_acc:.4f} ({int((y_val == pred).sum())}/{len(y_val)})")
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(cm_df.to_string())
    if "best_val_accuracy" in ckpt:
        print(f"\nCheckpoint reported best_val_accuracy (from training): {ckpt['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
