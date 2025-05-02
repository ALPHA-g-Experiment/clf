#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import polars as pl


def target_z(dataset, bins, min, max, normalize, alpha):
    z = pl.scan_parquet(dataset).select(z=pl.col("target").arr.get(2)).collect()["z"]
    min = z.min() if min is None else min
    max = z.max() if max is None else max

    plt.hist(
        z, bins=bins, range=(min, max), density=normalize, alpha=alpha, label=dataset
    )


def cloud_size(dataset, bins, min, max, normalize, alpha):
    size = (
        pl.scan_parquet(dataset)
        .select(size=pl.col("point_cloud").list.len())
        .collect()["size"]
    )
    min = size.min() if min is None else min
    max = size.max() if max is None else max

    plt.hist(
        size, bins=bins, range=(min, max), density=normalize, alpha=alpha, label=dataset
    )


def training(training_log):
    df = pl.read_csv(training_log)

    plt.plot(df["epoch"], df["training_loss"], label=f"Training ({training_log})")
    plt.plot(df["epoch"], df["validation_loss"], label=f"Validation ({training_log})")


def residuals(data, bins, min, max, normalize, alpha):
    res = pl.read_csv(data).select(res=pl.col("prediction") - pl.col("target"))["res"]
    min = res.min() if min is None else min
    max = res.max() if max is None else max

    plt.hist(
        res, bins=bins, range=(min, max), density=normalize, alpha=alpha, label=data
    )


def bias(data, num_slices, trim):
    df = pl.read_csv(data).select("target", res=pl.col("prediction") - pl.col("target"))
    breaks = pl.linear_space(
        df["target"].min(),
        df["target"].max(),
        num_samples=num_slices - 1,
        closed="none",
        eager=True,
    )
    df = (
        df.with_columns(bin=pl.col("target").cut(breaks))
        .group_by("bin")
        .agg(
            pl.mean("target"),
            pl.col("res").filter(
                pl.col("res").is_between(
                    pl.col("res").quantile(trim), pl.col("res").quantile(1 - trim)
                )
            ),
        )
        .select(
            z="target",
            mean=pl.col("res").list.mean(),
            std=pl.col("res").list.std().truediv(pl.col("res").list.len().sqrt()),
        )
    )

    plt.errorbar(df["z"], df["mean"], yerr=df["std"], label=data)


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

parser_target_z = subparsers.add_parser(
    "target-z",
    help="target z distribution (expects Parquet datasets)",
    parents=[parser_all, parser_hist],
)

parser_cloud_size = subparsers.add_parser(
    "cloud-size",
    help="point cloud size distribution (expects Parquet datasets)",
    parents=[parser_all, parser_hist],
)

parser_training = subparsers.add_parser(
    "training",
    help="training and validation loss (expects CSV training log)",
    parents=[parser_all],
)

parser_residuals = subparsers.add_parser(
    "residuals",
    help="residual distribution (expects CSV test results)",
    parents=[parser_all, parser_hist],
)

parser_bias = subparsers.add_parser(
    "bias",
    help="reconstruction bias (expects CSV test results)",
    parents=[parser_all],
)
parser_bias.add_argument("--slices", type=int, default=24, help="number of slices")
parser_bias.add_argument(
    "--trim", type=float, default=0.05, help="fraction to trim from each side"
)

args = parser.parse_args()

alpha = 1.0 if len(args.data) == 1 else 0.5
for dataset in args.data:
    match args.subcommand:
        case "target-z":
            target_z(dataset, args.bins, args.min, args.max, args.normalize, alpha)
            plt.xlabel("z [mm]")
            plt.ylabel("Count")

        case "cloud-size":
            cloud_size(dataset, args.bins, args.min, args.max, args.normalize, alpha)
            plt.xlabel("Number of points")
            plt.ylabel("Count")

        case "training":
            training(dataset)
            plt.xlabel("Epoch")
            plt.ylabel("Loss [a.u]")

        case "residuals":
            residuals(dataset, args.bins, args.min, args.max, args.normalize, alpha)
            plt.xlabel("Residual [mm]")
            plt.ylabel("Count")

        case "bias":
            bias(dataset, args.slices, args.trim)
            plt.xlabel("Target z [mm]")
            plt.ylabel("Residuals mean [mm]")

plt.legend()

if args.output:
    plt.savefig(args.output)
    print(f"Created `{args.output}`")
else:
    plt.show()
