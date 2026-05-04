"""
Train a 4-class audio classifier: wav2vec2-base features + small MLP.
Requires: torch, torchaudio, transformers, pandas, scikit-learn, numpy
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
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Fixed class order (matches thesis labels)
CLASS_NAMES = ["whisper", "room_noise", "distractor", "normal_speech"]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

METADATA_PATH = Path(__file__).resolve().parent / "metadata.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "whisper_classifier.pt"

MODEL_ID = "facebook/wav2vec2-base"
FEATURE_DIM = 768
HIDDEN1 = 512
HIDDEN2 = 256
DROPOUT = 0.1

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
SEED = 42
FEATURE_BATCH = 8  # WAV files per forward through wav2vec (memory)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_waveform(path: str, target_sr: int) -> torch.Tensor:
    """Load mono waveform float32 [T] at target_sr."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


@torch.inference_mode()
def extract_wav2vec_features(
    paths: list[str],
    device: torch.device,
    feat_extractor: Wav2Vec2FeatureExtractor,
    w2v: Wav2Vec2Model,
    target_sr: int,
) -> np.ndarray:
    """Mean-pooled last hidden state per file -> (N, 768)."""
    w2v.eval()
    out_list: list[np.ndarray] = []
    for start in range(0, len(paths), FEATURE_BATCH):
        batch_paths = paths[start : start + FEATURE_BATCH]
        waves = [load_waveform(p, target_sr) for p in batch_paths]
        # Pad for batching through processor
        inputs = feat_extractor(
            [w.numpy() for w in waves],
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = w2v(**inputs).last_hidden_state  # [B, T', 768]
        mask = inputs.get("attention_mask")
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = out.mean(dim=1)
        out_list.append(pooled.cpu().numpy())
    return np.concatenate(out_list, axis=0)


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
    if train:
        model.train()
    else:
        model.eval()
    preds: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    n = 0
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
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        labels.extend(yb.cpu().numpy().tolist())
    return total_loss / max(n, 1), np.array(labels), np.array(preds)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available; using CPU.", file=sys.stderr)

    df = pd.read_csv(METADATA_PATH)
    if "file_path" not in df.columns or "label" not in df.columns:
        raise SystemExit("metadata.csv must have columns: file_path, label")

    bad = set(df["label"].unique()) - set(CLASS_NAMES)
    if bad:
        raise SystemExit(f"Unknown labels in CSV (not in CLASS_NAMES): {bad}")

    paths = df["file_path"].astype(str).tolist()
    y = df["label"].map(LABEL_TO_IDX).astype(np.int64).values

    print(f"Loading {MODEL_ID} feature extractor on {device}...")
    feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    w2v = Wav2Vec2Model.from_pretrained(MODEL_ID).to(device)
    for p in w2v.parameters():
        p.requires_grad = False

    target_sr = feat_extractor.sampling_rate
    print(f"Extracting features for {len(paths)} files (sr={target_sr})...")
    X = extract_wav2vec_features(paths, device, feat_extractor, w2v, target_sr)
    del w2v
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    model = MLPClassifier(FEATURE_DIM, len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    print(f"\nTraining MLP for {EPOCHS} epochs (train={len(train_ds)}, val={len(val_ds)})...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        _, y_true_v, y_pred_v = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        model.eval()
        with torch.inference_mode():
            tr_logits = model(torch.from_numpy(X_train).float().to(device))
            train_acc = accuracy_score(y_train, tr_logits.argmax(1).cpu().numpy())
        val_acc = accuracy_score(y_true_v, y_pred_v)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    # Final metrics on validation set
    model.eval()
    with torch.inference_mode():
        logits_val = model(torch.from_numpy(X_val).float().to(device))
    y_pred_final = logits_val.argmax(1).cpu().numpy()

    acc_val = accuracy_score(y_val, y_pred_final)
    cm = confusion_matrix(
        y_val, y_pred_final, labels=list(range(len(CLASS_NAMES)))
    )
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

    print(f"\nFinal validation accuracy (argmax): {acc_val:.4f}")
    print("\nFinal confusion matrix (rows=true, columns=predicted):")
    print(cm_df.to_string())

    torch.save(
        {
            "mlp_state_dict": best_state,
            "classes": CLASS_NAMES,
            "label_to_idx": LABEL_TO_IDX,
            "wav2vec_model_id": MODEL_ID,
            "feature_dim": FEATURE_DIM,
            "mlp_hidden": (HIDDEN1, HIDDEN2),
            "best_val_accuracy": float(best_val_acc),
        },
        OUTPUT_PATH,
    )
    print(f"\nSaved best model to {OUTPUT_PATH} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
