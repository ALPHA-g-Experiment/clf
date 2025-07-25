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

    thresholds = np.linspace(0, 1, 100)
    true_positives = np.zeros_like(thresholds)
    false_positives = np.zeros_like(thresholds)

    for i, threshold in enumerate(thresholds):
        passed = df.filter(pl.col("prediction") >= threshold)

        true_positives[i] = passed.filter(pl.col("target") == 1.0).height
        false_positives[i] = passed.filter(pl.col("target") == 0.0).height

    plt.plot(false_positives / n_background, true_positives / n_signal, label=data)


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

plt.legend()

if args.output:
    plt.savefig(args.output)
    print(f"Created `{args.output}`")
else:
    plt.show()
