import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from timm.models import create_model
# Importing this registers LSNet models (lsnet_t, lsnet_s, lsnet_b, etc.) with timm
from model import build as _build  # noqa: F401

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
except ImportError as e:
    raise ImportError(
        "scikit-learn is required for eval_thyroid.py. "
        "Please install it with `pip install scikit-learn`."
    ) from e


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LSNet thyroid classifier on one or more test datasets."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint_best.pth (or similar).",
    )
    parser.add_argument(
        "--test-dirs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more test dataset roots. Each root should be an ImageFolder-style "
            "directory with subfolders for each class."
        ),
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Optional human-readable names for each test set (same length as --test-dirs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--pos-class-index",
        type=int,
        default=1,
        help=(
            "Index of the positive class for AUROC/AUPRC. "
            "For binary classification with two folders, usually 0 or 1."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=(
            "Optional path to a log file. If provided, all printed results will also "
            "be appended to this file."
        ),
    )
    return parser.parse_args()


def build_transform(input_size: int = 224) -> transforms.Compose:
    # Simple evaluation transform; should be consistent with training.
    size = int((256 / 224) * input_size)
    t = [
        transforms.Resize(size, interpolation=3),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(t)


def load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, int, str]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "args" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain 'args'. Make sure it was saved by main.py in this repo."
        )

    args_ckpt = checkpoint["args"]
    model_name = getattr(args_ckpt, "model", None)
    num_classes = getattr(args_ckpt, "nb_classes", None)

    if model_name is None or num_classes is None:
        raise RuntimeError(
            "Checkpoint 'args' does not contain 'model' or 'nb_classes'. "
            "Please ensure the model was trained with this repository's main.py."
        )

    distillation_type = getattr(args_ckpt, "distillation_type", "none")
    model = create_model(
        model_name,
        num_classes=num_classes,
        distillation=(distillation_type != "none"),
    )

    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model state dict with message: {msg}")

    model.to(device)
    model.eval()
    return model, num_classes, model_name


@torch.no_grad()
def evaluate_one_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pos_class_index: int,
) -> dict:
    all_probs: List[float] = []
    all_labels: List[int] = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(1)

        if outputs.size(1) == 1:
            probs_pos = torch.sigmoid(outputs.squeeze(1))
        else:
            probs = torch.softmax(outputs, dim=1)
            probs_pos = probs[:, pos_class_index]

        all_probs.append(probs_pos.detach().cpu().numpy())
        all_labels.append(targets.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            f"Expected binary labels for AUROC/AUPRC, got classes: {np.unique(y_true)}"
        )

    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Acc": acc,
        "Prec": prec,
        "Recall": rec,
        "F1": f1,
        "Specificity": specificity,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "N": int(len(y_true)),
    }


def main():
    args = parse_args()

    if args.names is not None and len(args.names) != len(args.test_dirs):
        raise ValueError("--names must be the same length as --test-dirs if provided.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_f = None
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True) if os.path.dirname(args.log_file) else None
        log_f = open(args.log_file, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        if log_f is not None:
            log_f.write(msg + "\n")
            log_f.flush()

    log(f"Using device: {device}")

    model, num_classes, model_name = load_model_from_checkpoint(
        args.checkpoint, device
    )
    log(f"Loaded model '{model_name}' with {num_classes} classes.")

    transform = build_transform(input_size=224)

    for idx, test_root in enumerate(args.test_dirs):
        if not os.path.isdir(test_root):
            log(f"[WARN] Test dir does not exist, skipped: {test_root}")
            continue

        name = args.names[idx] if args.names is not None else os.path.basename(
            os.path.normpath(test_root)
        )
        log(f"\n=== Evaluating on test set: {name} ===")
        log(f"Path: {test_root}")

        dataset = datasets.ImageFolder(test_root, transform=transform, loader=default_loader)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        log(f"Classes: {dataset.classes}")

        metrics = evaluate_one_dataset(
            model=model,
            dataloader=dataloader,
            device=device,
            pos_class_index=args.pos_class_index,
        )

        log(f"Samples: {metrics['N']}")
        log(f"TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']} TP={metrics['TP']}")
        log(
            f"AUROC={metrics['AUROC']:.4f}  AUPRC={metrics['AUPRC']:.4f}  "
            f"Acc={metrics['Acc']:.4f}  Prec={metrics['Prec']:.4f}  "
            f"Recall={metrics['Recall']:.4f}  F1={metrics['F1']:.4f}  "
            f"Specificity={metrics['Specificity']:.4f}"
        )

    if log_f is not None:
        log_f.close()


if __name__ == "__main__":
    main()

