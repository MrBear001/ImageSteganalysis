import argparse
import csv
import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


CURRENT_LR_RE = re.compile(r"Current learning rate:\s*([0-9.eE+-]+)")
TRAIN_AVG_LOSS_RE = re.compile(r"train Epoch:\s*(\d+)\s+avgLoss:\s*([0-9.]+)")
EVAL_RE = re.compile(
    r"(Valid|Test) set: Average loss:\s*([0-9.]+), Accuracy:\s*([0-9.]+)/([0-9]+)\s+\(([0-9.]+)%\)"
)
TOTAL_TIME_RE = re.compile(r"Total training time:\s*([0-9.]+)\s*seconds")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_experiment_name(log_path: str) -> str:
    name = os.path.splitext(os.path.basename(log_path))[0]
    return name.replace("-train", "")


def parse_log(log_path: str) -> Dict[str, object]:
    epochs: List[int] = []
    train_loss: List[float] = []
    valid_loss: List[float] = []
    valid_acc: List[float] = []
    test_loss: List[float] = []
    test_acc: List[float] = []
    learning_rates: List[Optional[float]] = []
    current_lr: Optional[float] = None
    total_time_seconds: Optional[float] = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            lr_match = CURRENT_LR_RE.search(line)
            if lr_match:
                current_lr = float(lr_match.group(1))
                continue

            train_match = TRAIN_AVG_LOSS_RE.search(line)
            if train_match:
                epochs.append(int(train_match.group(1)))
                train_loss.append(float(train_match.group(2)))
                learning_rates.append(current_lr)
                continue

            eval_match = EVAL_RE.search(line)
            if eval_match:
                split_name = eval_match.group(1).lower()
                avg_loss = float(eval_match.group(2))
                accuracy = float(eval_match.group(5))
                if split_name == "valid":
                    valid_loss.append(avg_loss)
                    valid_acc.append(accuracy)
                else:
                    test_loss.append(avg_loss)
                    test_acc.append(accuracy)
                continue

            time_match = TOTAL_TIME_RE.search(line)
            if time_match:
                total_time_seconds = float(time_match.group(1))

    if not epochs:
        raise ValueError(f"Failed to parse any epoch data from {log_path}")

    min_length = min(
        len(epochs),
        len(train_loss),
        len(valid_loss),
        len(valid_acc),
        len(test_loss),
        len(test_acc),
        len(learning_rates),
    )
    if min_length == 0:
        raise ValueError(f"Incomplete validation or test data in {log_path}")

    epochs = epochs[:min_length]
    train_loss = train_loss[:min_length]
    valid_loss = valid_loss[:min_length]
    valid_acc = valid_acc[:min_length]
    test_loss = test_loss[:min_length]
    test_acc = test_acc[:min_length]
    learning_rates = learning_rates[:min_length]

    best_valid_idx = max(range(min_length), key=lambda idx: valid_acc[idx])
    best_test_idx = max(range(min_length), key=lambda idx: test_acc[idx])

    return {
        "experiment": infer_experiment_name(log_path),
        "log_path": log_path,
        "epochs": epochs,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "learning_rates": learning_rates,
        "total_time_seconds": total_time_seconds,
        "best_valid_idx": best_valid_idx,
        "best_test_idx": best_test_idx,
    }


def save_epoch_csv(history: Dict[str, object], output_dir: str) -> str:
    csv_path = os.path.join(output_dir, f"{history['experiment']}_epoch_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "learning_rate",
                "train_loss",
                "valid_loss",
                "valid_accuracy",
                "test_loss",
                "test_accuracy",
            ]
        )
        for row in zip(
            history["epochs"],
            history["learning_rates"],
            history["train_loss"],
            history["valid_loss"],
            history["valid_acc"],
            history["test_loss"],
            history["test_acc"],
        ):
            writer.writerow(row)
    return csv_path


def plot_single_experiment(history: Dict[str, object], output_dir: str) -> str:
    experiment = history["experiment"]
    figure_path = os.path.join(output_dir, f"{experiment}_training_curves.png")
    epochs = history["epochs"]
    best_valid_idx = history["best_valid_idx"]
    best_test_idx = history["best_test_idx"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4")
    axes[0, 0].plot(epochs, history["valid_loss"], label="Valid Loss", color="#ff7f0e")
    axes[0, 0].plot(epochs, history["test_loss"], label="Test Loss", color="#2ca02c")
    axes[0, 0].set_title(f"{experiment} Loss Curves")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["valid_acc"], label="Valid Accuracy", color="#d62728")
    axes[0, 1].plot(epochs, history["test_acc"], label="Test Accuracy", color="#9467bd")
    axes[0, 1].scatter(
        epochs[best_valid_idx],
        history["valid_acc"][best_valid_idx],
        color="#d62728",
        s=60,
        label=f"Best Valid: {history['valid_acc'][best_valid_idx]:.2f}%",
    )
    axes[0, 1].scatter(
        epochs[best_test_idx],
        history["test_acc"][best_test_idx],
        color="#9467bd",
        s=60,
        label=f"Best Test: {history['test_acc'][best_test_idx]:.2f}%",
    )
    axes[0, 1].set_title(f"{experiment} Accuracy Curves")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history["learning_rates"], color="#8c564b")
    axes[1, 0].set_title(f"{experiment} Learning Rate Schedule")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis("off")
    summary_lines = [
        f"Experiment: {experiment}",
        f"Epochs: {len(epochs)}",
        f"Best Valid Accuracy: {history['valid_acc'][best_valid_idx]:.2f}% (Epoch {epochs[best_valid_idx]})",
        f"Best Test Accuracy: {history['test_acc'][best_test_idx]:.2f}% (Epoch {epochs[best_test_idx]})",
        f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%",
        f"Final Test Loss: {history['test_loss'][-1]:.4f}",
    ]
    if history["total_time_seconds"] is not None:
        hours = history["total_time_seconds"] / 3600.0
        summary_lines.append(f"Training Time: {history['total_time_seconds']:.2f}s ({hours:.2f}h)")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def save_summary_csv(histories: List[Dict[str, object]], output_dir: str) -> str:
    summary_path = os.path.join(output_dir, "experiment_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment",
                "epochs",
                "best_valid_epoch",
                "best_valid_accuracy",
                "best_test_epoch",
                "best_test_accuracy",
                "final_test_accuracy",
                "final_test_loss",
                "training_time_seconds",
            ]
        )
        for history in histories:
            best_valid_idx = history["best_valid_idx"]
            best_test_idx = history["best_test_idx"]
            epochs = history["epochs"]
            writer.writerow(
                [
                    history["experiment"],
                    len(epochs),
                    epochs[best_valid_idx],
                    history["valid_acc"][best_valid_idx],
                    epochs[best_test_idx],
                    history["test_acc"][best_test_idx],
                    history["test_acc"][-1],
                    history["test_loss"][-1],
                    history["total_time_seconds"],
                ]
            )
    return summary_path


def plot_comparison(histories: List[Dict[str, object]], output_dir: str) -> Optional[str]:
    if len(histories) < 2:
        return None

    figure_path = os.path.join(output_dir, "experiment_comparison.png")
    names = [history["experiment"] for history in histories]
    final_test_acc = [history["test_acc"][-1] for history in histories]
    best_test_acc = [history["test_acc"][history["best_test_idx"]] for history in histories]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(names, final_test_acc, color="#4c72b0")
    axes[0].set_title("Final Test Accuracy Comparison")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(names, best_test_acc, color="#55a868")
    axes[1].set_title("Best Test Accuracy Comparison")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def collect_default_logs(log_dir: str) -> List[str]:
    if not os.path.isdir(log_dir):
        return []
    return sorted(
        os.path.join(log_dir, filename)
        for filename in os.listdir(log_dir)
        if filename.lower().endswith(".log")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse training logs and export paper-ready figures and CSV summaries."
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        help="Training log files. If omitted, all .log files under --log-dir will be used.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="log",
        help="Directory that stores training logs when --logs is not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_outputs/log_plots",
        help="Directory used to save figures and CSV files.",
    )
    args = parser.parse_args()

    log_paths = args.logs or collect_default_logs(args.log_dir)
    if not log_paths:
        raise FileNotFoundError("No log files were found. Please provide --logs or check --log-dir.")

    ensure_dir(args.output_dir)

    histories: List[Dict[str, object]] = []
    for log_path in log_paths:
        history = parse_log(log_path)
        histories.append(history)
        csv_path = save_epoch_csv(history, args.output_dir)
        figure_path = plot_single_experiment(history, args.output_dir)
        print(f"[OK] Parsed log: {log_path}")
        print(f"     Epoch metrics: {csv_path}")
        print(f"     Training figure: {figure_path}")

    summary_csv = save_summary_csv(histories, args.output_dir)
    comparison_figure = plot_comparison(histories, args.output_dir)
    print(f"[OK] Summary table: {summary_csv}")
    if comparison_figure:
        print(f"[OK] Comparison figure: {comparison_figure}")


if __name__ == "__main__":
    main()
