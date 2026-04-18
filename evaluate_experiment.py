import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

import utils
from LWENet import lwenet


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_loader(cover_dir: str, stego_dir: str, batch_size: int, use_cuda: bool):
    dataset = utils.DatasetPair(cover_dir, stego_dir, utils.ToTensor())
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )


def evaluate(args) -> Dict[str, object]:
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader = build_loader(args.cover_dir, args.stego_dir, args.batch_size, use_cuda)

    model = lwenet().to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_scores: List[float] = []
    all_cover_probs: List[float] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            pair_count = images.shape[0]
            images = images.view(pair_count * 2, 1, 256, 256)
            labels = labels.view(pair_count * 2)

            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            loss = F.nll_loss(torch.log(probabilities.clamp_min(1e-12)), labels, reduction="sum")

            total_loss += loss.item()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predictions.cpu().tolist())
            all_scores.extend(probabilities[:, 1].cpu().tolist())
            all_cover_probs.extend(probabilities[:, 0].cpu().tolist())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    scores_np = np.array(all_scores)

    accuracy = accuracy_score(labels_np, preds_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(labels_np, scores_np, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return {
        "device": str(device),
        "sample_count": int(labels_np.shape[0]),
        "average_loss": total_loss / max(1, labels_np.shape[0]),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(roc_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": classification_report(
            labels_np,
            preds_np,
            target_names=["cover", "stego"],
            zero_division=0,
            output_dict=True,
        ),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "labels": labels_np.tolist(),
        "predictions": preds_np.tolist(),
        "stego_probabilities": scores_np.tolist(),
        "cover_probabilities": all_cover_probs,
    }


def save_metrics_json(metrics: Dict[str, object], output_dir: str) -> str:
    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    payload = dict(metrics)
    payload.pop("labels")
    payload.pop("predictions")
    payload.pop("stego_probabilities")
    payload.pop("cover_probabilities")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return metrics_path


def save_prediction_csv(metrics: Dict[str, object], output_dir: str) -> str:
    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "true_label", "predicted_label", "cover_probability", "stego_probability"])
        for idx, row in enumerate(
            zip(
                metrics["labels"],
                metrics["predictions"],
                metrics["cover_probabilities"],
                metrics["stego_probabilities"],
            )
        ):
            writer.writerow([idx, row[0], row[1], row[2], row[3]])
    return csv_path


def plot_roc(metrics: Dict[str, object], output_dir: str) -> str:
    figure_path = os.path.join(output_dir, "roc_curve.png")
    fpr = np.array(metrics["roc_curve"]["fpr"])
    tpr = np.array(metrics["roc_curve"]["tpr"])

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#d62728", lw=2, label=f"AUC = {metrics['auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", lw=1.5, label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    return figure_path


def plot_confusion_matrix(metrics: Dict[str, object], output_dir: str) -> str:
    figure_path = os.path.join(output_dir, "confusion_matrix.png")
    cm = np.array(
        [
            [metrics["confusion_matrix"]["tn"], metrics["confusion_matrix"]["fp"]],
            [metrics["confusion_matrix"]["fn"], metrics["confusion_matrix"]["tp"]],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    plt.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=["Pred Cover", "Pred Stego"])
    ax.set_yticks([0, 1], labels=["True Cover", "True Stego"])
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=12)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_probability_histogram(metrics: Dict[str, object], output_dir: str) -> str:
    figure_path = os.path.join(output_dir, "probability_histogram.png")
    labels = np.array(metrics["labels"])
    stego_probs = np.array(metrics["stego_probabilities"])

    plt.figure(figsize=(7, 6))
    plt.hist(stego_probs[labels == 0], bins=30, alpha=0.7, label="Cover", color="#1f77b4")
    plt.hist(stego_probs[labels == 1], bins=30, alpha=0.7, label="Stego", color="#ff7f0e")
    plt.xlabel("Predicted Probability of Stego")
    plt.ylabel("Number of Images")
    plt.title("Probability Distribution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    return figure_path


def print_summary(metrics: Dict[str, object]) -> None:
    print("=" * 54)
    print("LWENet Evaluation Summary")
    print("=" * 54)
    print(f"Device: {metrics['device']}")
    print(f"Samples: {metrics['sample_count']}")
    print(f"Average Loss: {metrics['average_loss']:.6f}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.4f}%")
    print(f"Precision: {metrics['precision'] * 100:.4f}%")
    print(f"Recall: {metrics['recall'] * 100:.4f}%")
    print(f"F1-score: {metrics['f1_score'] * 100:.4f}%")
    print(f"AUC: {metrics['auc']:.6f}")
    print(
        "Confusion Matrix: "
        f"TN={metrics['confusion_matrix']['tn']}, "
        f"FP={metrics['confusion_matrix']['fp']}, "
        f"FN={metrics['confusion_matrix']['fn']}, "
        f"TP={metrics['confusion_matrix']['tp']}"
    )
    print("=" * 54)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LWENet checkpoint and export paper-ready metrics and plots."
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pkl)")
    parser.add_argument("--cover-dir", type=str, required=True, help="Path to test cover images")
    parser.add_argument("--stego-dir", type=str, required=True, help="Path to test stego images")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Force CPU mode")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_outputs/model_eval",
        help="Directory used to save metrics and plots.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    if not os.path.isdir(args.cover_dir):
        raise FileNotFoundError(f"Cover directory not found: {args.cover_dir}")
    if not os.path.isdir(args.stego_dir):
        raise FileNotFoundError(f"Stego directory not found: {args.stego_dir}")

    ensure_dir(args.output_dir)
    metrics = evaluate(args)
    print_summary(metrics)

    metrics_path = save_metrics_json(metrics, args.output_dir)
    predictions_path = save_prediction_csv(metrics, args.output_dir)
    roc_path = plot_roc(metrics, args.output_dir)
    cm_path = plot_confusion_matrix(metrics, args.output_dir)
    hist_path = plot_probability_histogram(metrics, args.output_dir)

    print(f"[OK] Metrics JSON: {metrics_path}")
    print(f"[OK] Prediction CSV: {predictions_path}")
    print(f"[OK] ROC figure: {roc_path}")
    print(f"[OK] Confusion matrix: {cm_path}")
    print(f"[OK] Probability histogram: {hist_path}")


if __name__ == "__main__":
    main()
