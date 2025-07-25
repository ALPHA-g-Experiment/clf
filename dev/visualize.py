#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def scores(data, bins, min, max, normalize, alpha):
    df = pl.read_csv(data)
    min = df["prediction"].min() if min is None else min
    max = df["prediction"].max() if max is None else max
    bins = np.linspace(min, max, bins + 1)

    signal = df.filter(pl.col("target") == 1.0)["prediction"]
    background = df.filter(pl.col("target") == 0.0)["prediction"]

    plt.hist(
        background,
        bins=bins,
        density=normalize,
        alpha=alpha,
        label=f"Background ({data})",
    )
    plt.hist(
        signal, bins=bins, density=normalize, alpha=alpha, label=f"Signal ({data})"
    )


def roc(data):
    df = pl.read_csv(data)

    n_signal = df.filter(pl.col("target") == 1.0).height
    n_background = df.filter(pl.col("target") == 0.0).height

    thresholds = np.linspace(0, 1, 1000)
    # One extra point to force (0, 0). The score saturates, so even with a
    # threshold of 1.0, some predictions will satisfy it.
    true_positives = np.zeros(len(thresholds) + 1)
    false_positives = np.zeros_like(true_positives)

    for i, threshold in enumerate(thresholds):
        passed = df.filter(pl.col("prediction") >= threshold)

        true_positives[i] = passed.filter(pl.col("target") == 1.0).height
        false_positives[i] = passed.filter(pl.col("target") == 0.0).height

    fpr = false_positives / n_background
    tpr = true_positives / n_signal
    auc = abs(np.trapz(tpr, fpr))

    plt.plot(fpr, tpr, label=f"{data} (AUC = {auc:.4f})")


def confusion_matrix(data, threshold):
    df = pl.read_csv(data)

    signal_df = df.filter(pl.col("target") == 1.0)
    background_df = df.filter(pl.col("target") == 0.0)

    true_positive = signal_df.filter(pl.col("prediction") >= threshold).height
    false_negative = signal_df.filter(pl.col("prediction") < threshold).height
    false_positive = background_df.filter(pl.col("prediction") >= threshold).height
    true_negative = background_df.filter(pl.col("prediction") < threshold).height

    matrix = np.array(
        [[true_positive, false_negative], [false_positive, true_negative]]
    )
    plt.imshow(matrix, cmap="Blues", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Confusion Matrix (Threshold = {threshold})")
    plt.xticks([0, 1], ["Predicted Signal", "Predicted Background"])
    plt.yticks([0, 1], ["Actual Signal", "Actual Background"])
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{matrix[i, j]}",
                ha="center",
                va="center",
                color="white",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
            )


parser = argparse.ArgumentParser(description="Data Visualization")
subparsers = parser.add_subparsers(dest="subcommand", required=True)

parser_all = argparse.ArgumentParser(add_help=False)
parser_all.add_argument(
    "data",
    nargs="+",
    help="path to a data file (expected format depends on the subcommand)",
)
parser_all.add_argument("--output", help="write output to `OUTPUT`")

parser_hist = argparse.ArgumentParser(add_help=False)
parser_hist.add_argument("--bins", type=int, default=100, help="number of bins")
parser_hist.add_argument("--max", type=float, help="maximum value")
parser_hist.add_argument("--min", type=float, help="minimum value")
parser_hist.add_argument("--normalize", action="store_true", help="normalize data")

parser_scores = subparsers.add_parser(
    "scores",
    help="score distribution (expects CSV test results)",
    parents=[parser_all, parser_hist],
)

parser_roc = subparsers.add_parser(
    "roc",
    help="ROC curve (expects CSV test results)",
    parents=[parser_all],
)

parser_confusion_matrix = subparsers.add_parser(
    "confusion-matrix",
    help="confusion matrix (expects CSV test results)",
    parents=[parser_all],
)
parser_confusion_matrix.add_argument(
    "--threshold", type=float, default=0.5, help="classification threshold"
)

args = parser.parse_args()

alpha = 1.0 if len(args.data) == 1 else 0.5
for dataset in args.data:
    match args.subcommand:
        case "scores":
            scores(dataset, args.bins, args.min, args.max, args.normalize, 0.6)
            plt.xlabel("Normalized logit")
            plt.ylabel("Count")
        case "roc":
            roc(dataset)
            plt.xlabel("1 - Specificity")
            plt.ylabel("Sensitivity")
        case "confusion-matrix":
            confusion_matrix(dataset, args.threshold)

if args.subcommand != "confusion-matrix":
    plt.legend()

if args.output:
    plt.savefig(args.output)
    print(f"Created `{args.output}`")
else:
    plt.show()
