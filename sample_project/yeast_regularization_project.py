"""Yeast Protein Localization Project

End-to-end multiclass classification experiments for the Yeast dataset.
This script trains and compares four MLP variants:
    1) No regularization
    2) Dropout
    3) L2 weight decay
    4) Dropout + L2

It produces:
    - Training/validation loss curves
    - A validation-loss comparison plot
    - A final test-performance table
    - Saved model checkpoints

Expected input:
    yeast.csv with a target column named `name` and 8 numeric feature columns.

Run:
    python yeast_regularization_project.py --data yeast.csv
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple  # Python 3.9+ could use built-in dict, list, tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sk_autod import diagnose


DEFAULT_TARGET_COL = "name"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "yeast.csv"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 100  # Reduced from 1000 for faster demo runs
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_DIM = 32
DEFAULT_LR = 1e-2
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.5


@dataclass
class DatasetBundle:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    class_names: np.ndarray
    input_dim: int
    output_dim: int


@dataclass
class TrainHistory:
    train_losses: List[float]
    val_losses: List[float]


@dataclass
class ModelResult:
    name: str
    test_accuracy: float
    macro_f1: float
    train_history: TrainHistory
    model: nn.Module


def log_step(message: str) -> None:
    print(f"[SK-AutoD Demo] {message}")


def diagnose_history(name: str, history: TrainHistory, output_dir: Path) -> None:
    if len(history.train_losses) < 5 or len(history.val_losses) < 5:
        print(f"SK-AutoD diagnosis for {name} skipped: need at least 5 epochs of history.")
        return

    report = diagnose(history.train_losses, history.val_losses)
    diagnosis_text = report.summary()
    print(f"\nSK-AutoD diagnosis for {name}:")
    print(diagnosis_text)
    (output_dir / f"curve_diagnosis_{name.lower().replace(' ', '_').replace('+', 'plus')}.txt").write_text(
        diagnosis_text,
        encoding="utf-8",
    )


def resolve_data_path(data_path: str) -> Path:
    candidate = Path(data_path).expanduser()
    if candidate.exists():
        return candidate

    script_candidate = SCRIPT_DIR / candidate
    if script_candidate.exists():
        return script_candidate

    return candidate


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class YeastMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_and_prepare_data(data_path: str, target_col: str = DEFAULT_TARGET_COL) -> DatasetBundle:
    resolved_data_path = resolve_data_path(data_path)
    log_step(f"Loading dataset from {resolved_data_path}")
    df = pd.read_csv(resolved_data_path)

    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in {resolved_data_path}. Found columns: {list(df.columns)}")

    log_step(f"Found target column '{target_col}' and {len(df.columns) - 1} feature columns")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.30,
        random_state=DEFAULT_SEED,
        stratify=y_encoded,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=DEFAULT_SEED,
        stratify=y_temp,
    )

    bundle = DatasetBundle(
        X_train=torch.tensor(X_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.long),
        X_val=torch.tensor(X_val, dtype=torch.float32),
        y_val=torch.tensor(y_val, dtype=torch.long),
        X_test=torch.tensor(X_test, dtype=torch.float32),
        y_test=torch.tensor(y_test, dtype=torch.long),
        class_names=class_names,
        input_dim=X_train.shape[1],
        output_dim=len(class_names),
    )
    log_step("Dataset split into train, validation, and test sets")
    return bundle


def make_dataloader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    logits = model(X)
    preds = torch.argmax(logits, dim=1)
    y_true = y.cpu().numpy()
    y_pred = preds.cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return acc, macro_f1, y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> TrainHistory:
    device = device or torch.device("cpu")
    model.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    log_step(f"Training on {device.type.upper()} for {epochs} epochs")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(epochs):
        should_report_epoch = epoch < 5 or (epoch + 1) % 10 == 0 or (epoch + 1) == epochs
        if should_report_epoch:
            print(f"  Epoch {epoch + 1}/{epochs}: training...")

        model.train()
        running_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_train_loss = running_loss / total_samples
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:4d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if should_report_epoch:
            print(f"    train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f}")

    return TrainHistory(train_losses=train_losses, val_losses=val_losses)


def plot_loss_curves(histories: Dict[str, TrainHistory], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for ax, (name, hist) in zip(axs, histories.items()):
        ax.plot(hist.train_losses, label="Training Loss")
        ax.plot(hist.val_losses, label="Validation Loss")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

    plt.suptitle("Comparison of Regularization Strategies", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = output_dir / "regularization_loss_comparison.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_validation_overlay(histories: Dict[str, TrainHistory], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))
    for name, hist in histories.items():
        plt.plot(hist.val_losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Across Regularization Techniques")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = output_dir / "validation_loss_overlay.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    bundle: DatasetBundle,
    epochs: int,
    lr: float,
    weight_decay: float,
    output_dir: Path,
    device: torch.device,
    save_plots: bool,
) -> ModelResult:
    log_step(f"Starting experiment: {name}")
    history = train_model(
        model=model,
        train_loader=train_loader,
        X_val=bundle.X_val,
        y_val=bundle.y_val,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    diagnose_history(name, history, output_dir)

    log_step(f"Evaluating experiment: {name}")
    test_accuracy, macro_f1, y_true, y_pred = evaluate(model.to(device), bundle.X_test.to(device), bundle.y_test.to(device))
    print(f"Test Accuracy ({name}): {test_accuracy:.4f}")
    print(f"Macro F1 ({name}): {macro_f1:.4f}")

    safe_name = name.lower().replace(" ", "_").replace("+", "plus")
    log_step(f"Saving model checkpoint for {name}")
    torch.save(model.state_dict(), output_dir / f"{safe_name}.pth")
    if save_plots:
        log_step(f"Saving confusion matrix for {name}")
        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=bundle.class_names,
            title=f"Confusion Matrix – {name}",
            output_path=output_dir / f"confusion_matrix_{safe_name}.png",
        )

    return ModelResult(
        name=name,
        test_accuracy=test_accuracy,
        macro_f1=macro_f1,
        train_history=history,
        model=model,
    )


def build_results_table(results: List[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": [r.name for r in results],
            "Test Accuracy": [r.test_accuracy for r in results],
            "Macro F1 Score": [r.macro_f1 for r in results],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Yeast multiclass classification with regularization.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to dataset CSV")
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL, help="Name of target column")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for plots and checkpoints")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Hidden layer width")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="L2 regularization strength")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout probability")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--save-plots", action="store_true", help="Generate matplotlib plots and confusion matrices")
    args = parser.parse_args()

    log_step("Initializing run")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Output directory ready at {output_dir}")

    log_step("Preparing dataset")
    bundle = load_and_prepare_data(args.data, target_col=args.target_col)
    print(f"Train set: {bundle.X_train.shape}")
    print(f"Validation set: {bundle.X_val.shape}")
    print(f"Test set: {bundle.X_test.shape}")
    print(f"Classes ({bundle.output_dim}): {list(bundle.class_names)}")

    log_step("Building training dataloader")
    train_loader = make_dataloader(bundle.X_train, bundle.y_train, batch_size=args.batch_size, shuffle=True)

    log_step("Creating model experiments")
    experiments = [
        ("No Reg (M1)", YeastMLP(bundle.input_dim, args.hidden_dim, bundle.output_dim, dropout_rate=0.0), 0.0),
        ("Dropout (M2)", YeastMLP(bundle.input_dim, args.hidden_dim, bundle.output_dim, dropout_rate=args.dropout), 0.0),
        ("L2 (M3)", YeastMLP(bundle.input_dim, args.hidden_dim, bundle.output_dim, dropout_rate=0.0), args.weight_decay),
        ("Dropout + L2 (M4)", YeastMLP(bundle.input_dim, args.hidden_dim, bundle.output_dim, dropout_rate=args.dropout), args.weight_decay),
    ]

    results: List[ModelResult] = []
    histories: Dict[str, TrainHistory] = {}

    for name, model, wd in experiments:
        log_step(f"Running experiment '{name}'")
        result = run_experiment(
            name=name,
            model=model,
            train_loader=train_loader,
            bundle=bundle,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=wd,
            output_dir=output_dir,
            device=device,
            save_plots=args.save_plots,
        )
        results.append(result)
        histories[name] = result.train_history

    if args.save_plots:
        log_step("Generating loss curve plots")
        plot_loss_curves(histories, output_dir)
        log_step("Generating validation overlay plot")
        plot_validation_overlay(histories, output_dir)

    log_step("Compiling final results table")
    results_df = build_results_table(results)
    print("\nFinal Model Performance on Test Data:")
    print(results_df.to_string(index=False))

    log_step("Saving summary outputs")
    results_df.to_csv(output_dir / "test_performance_summary.csv", index=False)
    (output_dir / "class_names.json").write_text(json.dumps(list(bundle.class_names), indent=2))

    best = results_df.sort_values(["Macro F1 Score", "Test Accuracy"], ascending=False).iloc[0]
    print(f"\nBest model by Macro F1: {best['Model']} (F1={best['Macro F1 Score']:.4f}, Acc={best['Test Accuracy']:.4f})")
    log_step("Run complete")


if __name__ == "__main__":
    main()
