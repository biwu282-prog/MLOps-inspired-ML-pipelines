from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "deep_learning" / "data" / "animals10"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "deep_learning" / "training"
MODEL_NAMES = ("resnet18", "vgg16")


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    patience: int = 5
    seed: int = 42
    num_workers: int = 0
    dog_limit: int = 2000
    spider_limit: int = 2000
    freeze_backbone: bool = True
    models: str = "resnet18,vgg16"
    dry_run: bool = False


class TransformSubset(Dataset):
    def __init__(self, base: datasets.ImageFolder, indices: Iterable[int], transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.targets = [base.targets[i] for i in self.indices]
        self.samples = [base.samples[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        base_index = self.indices[item]
        path, label = self.base.samples[base_index]
        image = self.base.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train pretrained ResNet18 and VGG16 on balanced Animal-10 data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dog-limit", type=int, default=2000)
    parser.add_argument("--spider-limit", type=int, default=2000)
    parser.add_argument("--models", default="resnet18,vgg16", help="Comma-separated: resnet18,vgg16")
    parser.add_argument(
        "--fine-tune-all",
        action="store_true",
        help="Train all pretrained layers. Default trains only the classification head.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Prepare data summaries without training models.")
    args = parser.parse_args()

    return TrainConfig(
        data_dir=str(args.data_dir.resolve()),
        output_dir=str(args.output_dir.resolve()),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        dog_limit=args.dog_limit,
        spider_limit=args.spider_limit,
        freeze_backbone=not args.fine_tune_all,
        models=args.models,
        dry_run=args.dry_run,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def verify_image(path: str) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def load_clean_dataset(data_dir: Path, output_dir: Path) -> datasets.ImageFolder:
    dataset = datasets.ImageFolder(root=data_dir)
    clean_samples = []
    invalid_paths = []

    for path, target in dataset.samples:
        if verify_image(path):
            clean_samples.append((path, target))
        else:
            invalid_paths.append(path)

    dataset.samples = clean_samples
    dataset.imgs = clean_samples
    dataset.targets = [target for _, target in clean_samples]

    if invalid_paths:
        with (output_dir / "invalid_images_v5.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(invalid_paths))

    return dataset


def downsample_dog_spider(dataset: datasets.ImageFolder, config: TrainConfig) -> dict:
    rng = random.Random(config.seed)
    limits = {"dog": config.dog_limit, "spider": config.spider_limit}
    class_to_samples: dict[int, list[tuple[str, int]]] = {}
    for sample in dataset.samples:
        class_to_samples.setdefault(sample[1], []).append(sample)

    selected_samples = []
    summary = {}
    for class_name, class_index in dataset.class_to_idx.items():
        samples = class_to_samples.get(class_index, [])
        original_count = len(samples)
        limit = limits.get(class_name)
        if limit is not None and original_count > limit:
            samples = samples.copy()
            rng.shuffle(samples)
            samples = samples[:limit]
        selected_samples.extend(samples)
        summary[class_name] = {"original": original_count, "used": len(samples)}

    selected_samples.sort(key=lambda item: (item[1], item[0]))
    dataset.samples = selected_samples
    dataset.imgs = selected_samples
    dataset.targets = [target for _, target in selected_samples]
    return summary


def split_indices(targets: list[int], config: TrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(targets))
    labels = np.asarray(targets)
    holdout_fraction = config.val_fraction + config.test_fraction
    if holdout_fraction <= 0 or holdout_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be between 0 and 1.")

    first = StratifiedShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=config.seed)
    train_idx, holdout_idx = next(first.split(indices, labels))

    relative_test_fraction = config.test_fraction / holdout_fraction
    second = StratifiedShuffleSplit(n_splits=1, test_size=relative_test_fraction, random_state=config.seed + 1)
    val_relative_idx, test_relative_idx = next(second.split(holdout_idx, labels[holdout_idx]))
    return train_idx, holdout_idx[val_relative_idx], holdout_idx[test_relative_idx]


def build_transforms(image_size: int):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_model(model_name: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        if freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    if not train:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def predict(model, loader, device) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    paths = []
    labels_all = []
    preds_all = []

    with torch.no_grad():
        for images, labels, batch_paths in loader:
            images = images.to(device)
            outputs = model(images)
            preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())
            paths.extend(batch_paths)

    return paths, np.asarray(labels_all), np.asarray(preds_all)


def load_weights(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def save_history(history: list[dict], model_dir: Path, model_name: str) -> None:
    with (model_dir / f"history_{model_name}_v5.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} Loss Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train Accuracy")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} Accuracy Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(model_dir / f"training_curves_{model_name}_v5.png", dpi=180)
    plt.close(fig)


def save_model_outputs(
    model_name: str,
    paths: list[str],
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    model_dir: Path,
) -> dict:
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
    (model_dir / f"classification_report_{model_name}_v5.txt").write_text(report, encoding="utf-8")

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    per_class_rows = []
    for index, class_name in enumerate(class_names):
        per_class_rows.append(
            {
                "class": class_name,
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "accuracy": recall[index],
                "support": int(support[index]),
            }
        )

    with (model_dir / f"per_class_metrics_{model_name}_v5.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_class_rows)

    prediction_rows = []
    for path, label, pred in zip(paths, labels, preds):
        prediction_rows.append(
            {
                "path": path,
                "true_label": class_names[int(label)],
                "predicted_label": class_names[int(pred)],
                "correct": int(label == pred),
            }
        )

    with (model_dir / f"test_predictions_{model_name}_v5.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)

    cm = confusion_matrix(labels, preds, labels=np.arange(len(class_names)))
    np.savetxt(model_dir / f"confusion_matrix_{model_name}_v5.csv", cm, fmt="%d", delimiter=",")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(model_dir / f"confusion_matrix_{model_name}_v5.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_names, recall * 100)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{model_name} Per-Class Accuracy")
    fig.tight_layout()
    fig.savefig(model_dir / f"per_class_accuracy_{model_name}_v5.png", dpi=180)
    plt.close(fig)

    return {
        "model": model_name,
        "test_accuracy": accuracy_score(labels, preds),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
    }


def save_comparison(outputs: list[dict], output_dir: Path) -> None:
    with (output_dir / "model_comparison_v5.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=outputs[0].keys())
        writer.writeheader()
        writer.writerows(outputs)

    labels = [row["model"] for row in outputs]
    metrics = ["test_accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, metric in enumerate(metrics):
        values = [row[metric] for row in outputs]
        ax.bar(x + (offset - 1) * width, values, width, label=metric)

    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("VGG16 vs ResNet18 Test Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_v5.png", dpi=180)
    plt.close(fig)


def train_one_model(
    model_name: str,
    config: TrainConfig,
    class_names: list[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> dict:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_name, len(class_names), config.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    history = []
    best_val_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_model_path = model_dir / f"best_{model_name}_v5.pth"
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_accuracy)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"{model_name} | Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            print(f"{model_name} early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    save_history(history, model_dir, model_name)
    model.load_state_dict(load_weights(best_model_path, device))
    paths, labels, preds = predict(model, test_loader, device)
    summary = save_model_outputs(model_name, paths, labels, preds, class_names, model_dir)
    summary.update(
        {
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "training_seconds": time.time() - start,
        }
    )
    (model_dir / f"metrics_{model_name}_v5.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    config = parse_args()
    requested_models = [name.strip().lower() for name in config.models.split(",") if name.strip()]
    unsupported = sorted(set(requested_models) - set(MODEL_NAMES))
    if unsupported:
        raise ValueError(f"Unsupported model names: {unsupported}. Supported: {MODEL_NAMES}")

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_clean_dataset(Path(config.data_dir), output_dir)
    balance_summary = downsample_dog_spider(dataset, config)
    train_idx, val_idx, test_idx = split_indices(dataset.targets, config)
    train_transform, eval_transform = build_transforms(config.image_size)

    train_dataset = TransformSubset(dataset, train_idx, train_transform)
    val_dataset = TransformSubset(dataset, val_idx, eval_transform)
    test_dataset = TransformSubset(dataset, test_idx, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    split_summary = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }
    metadata = {
        "config": asdict(config),
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "balance_summary": balance_summary,
        "split_sizes": split_summary,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }
    (output_dir / "training_metadata_v5.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with (output_dir / "balanced_class_counts_v5.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "original", "used"])
        writer.writeheader()
        for class_name, counts in balance_summary.items():
            writer.writerow({"class": class_name, **counts})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Split sizes: {split_summary}")
    print("Balanced class counts:")
    for class_name, counts in balance_summary.items():
        print(f"  {class_name}: {counts['used']} / {counts['original']}")

    if config.dry_run:
        print(f"Dry run complete. Metadata saved to: {output_dir}")
        return

    summaries = []
    for model_name in requested_models:
        summaries.append(
            train_one_model(
                model_name,
                config,
                dataset.classes,
                train_loader,
                val_loader,
                test_loader,
                device,
                output_dir,
            )
        )

    save_comparison(summaries, output_dir)
    (output_dir / "metrics_summary_v5.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
