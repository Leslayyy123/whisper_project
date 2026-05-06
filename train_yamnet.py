"""
Train a 4-class audio classifier using YAMNet embeddings + MLP.

Dataset source: metadata.csv (file_path, label)
Classes: whisper, room_noise, distractor, normal_speech
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES = ["whisper", "room_noise", "distractor", "normal_speech"]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

INPUT_SAMPLE_RATE = 16000
YAMNET_EMBED_DIM = 1024
EXTRA_FEATURES_DIM = 3
EMBED_DIM = YAMNET_EMBED_DIM + EXTRA_FEATURES_DIM
HIDDEN1 = 512
HIDDEN2 = 256
DROPOUT = 0.1
EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3
SEED = 42

METADATA_PATH = Path(__file__).resolve().parent / "metadata.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "yamnet_classifier.pt"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_waveform(path: str) -> np.ndarray:
    """Load mono waveform as float32 numpy array at 16kHz."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != INPUT_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, INPUT_SAMPLE_RATE)
    return wav.squeeze(0).numpy().astype(np.float32)


def extract_acoustic_features(waveform: np.ndarray) -> np.ndarray:
    """
    Extract handcrafted acoustic features:
    - fricative_ratio: energy in 3000-8000 Hz / total energy
    - rms: normalized root-mean-square level
    - peak: normalized peak absolute level
    """
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


@torch.inference_mode()
def extract_yamnet_embeddings(paths: list[str], yamnet_model) -> np.ndarray:
    """Extract [YAMNet(1024) + acoustic(3)] -> (N, 1027)."""
    feats: list[np.ndarray] = []
    for i, path in enumerate(paths, start=1):
        waveform = load_waveform(path)
        _, embeddings, _ = yamnet_model(waveform)
        # embeddings: [num_patches, 1024], average across time patches
        emb = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
        acoustic = extract_acoustic_features(waveform)
        feat = np.concatenate([emb, acoustic], axis=0)
        feats.append(feat)
        if i % 25 == 0 or i == len(paths):
            print(f"Extracted YAMNet embeddings: {i}/{len(paths)}")
    return np.stack(feats, axis=0)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_epoch(
    model: MLPClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(mode=train)
    total_loss = 0.0
    n = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return total_loss / max(n, 1), np.array(y_true), np.array(y_pred)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available; training classifier on CPU.", file=sys.stderr)

    if not METADATA_PATH.exists():
        raise SystemExit(f"metadata.csv not found at: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)
    if "file_path" not in df.columns or "label" not in df.columns:
        raise SystemExit("metadata.csv must contain columns: file_path, label")

    bad_labels = set(df["label"].unique()) - set(CLASS_NAMES)
    if bad_labels:
        raise SystemExit(f"Unknown labels in metadata.csv: {bad_labels}")

    paths = df["file_path"].astype(str).tolist()
    y = df["label"].map(LABEL_TO_IDX).astype(np.int64).to_numpy()
    print(f"Loaded {len(paths)} files from metadata.csv")

    print("Loading YAMNet from TensorFlow Hub...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    print("Extracting embeddings...")
    X = extract_yamnet_embeddings(paths, yamnet_model)
    if X.shape[1] != EMBED_DIM:
        raise SystemExit(f"Unexpected embedding dimension: {X.shape[1]} (expected {EMBED_DIM})")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MLPClassifier(EMBED_DIM, len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    print(f"\nTraining MLP for {EPOCHS} epochs (train={len(train_ds)}, val={len(val_ds)})...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss, y_true_tr, y_pred_tr = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        _, y_true_v, y_pred_v = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )

        train_acc = accuracy_score(y_true_tr, y_pred_tr)
        val_acc = accuracy_score(y_true_v, y_pred_v)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    model.eval()
    with torch.inference_mode():
        val_logits = model(torch.from_numpy(X_val).float().to(device))
    y_pred_final = val_logits.argmax(dim=1).cpu().numpy()

    final_acc = accuracy_score(y_val, y_pred_final)
    cm = confusion_matrix(y_val, y_pred_final, labels=list(range(len(CLASS_NAMES))))
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

    print(f"\nFinal validation accuracy: {final_acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    torch.save(
        {
            "mlp_state_dict": best_state,
            "classes": CLASS_NAMES,
            "label_to_idx": LABEL_TO_IDX,
            "embedding_model": "google/yamnet/1",
            "input_sample_rate": INPUT_SAMPLE_RATE,
            "embedding_dim": EMBED_DIM,
            "mlp_hidden": (HIDDEN1, HIDDEN2),
            "best_val_accuracy": float(best_val_acc),
        },
        OUTPUT_PATH,
    )
    print(f"\nSaved best model to {OUTPUT_PATH} (best_val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
