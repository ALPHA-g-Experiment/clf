#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import polars as pl


def target_z(dataset, output, bins):
    df = pl.scan_parquet(dataset).select(z=pl.col("target").arr.get(2)).collect()

    plt.hist(df["z"], bins=bins)
    plt.xlabel("z [mm]")
    plt.ylabel("count")

    if output:
        plt.savefig(output)
    else:
        plt.show()


def cloud_size(dataset, output, bins):
    df = (
        pl.scan_parquet(dataset).select(size=pl.col("point_cloud").list.len()).collect()
    )

    plt.hist(df["size"], bins=bins)
    plt.xlabel("Number of points")
    plt.ylabel("count")

    if output:
        plt.savefig(output)
    else:
        plt.show()


parser = argparse.ArgumentParser(
    description="Dataset visualization",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("dataset", help="path to the Parquet dataset")
parser.add_argument("--output", help="write output to `OUTPUT`")
subparsers = parser.add_subparsers(
    dest="subcommand", required=True, help="visualization"
)

parser_hist = argparse.ArgumentParser(add_help=False)
parser_hist.add_argument("--bins", type=int, default=100, help="number of bins")

parser_target_z = subparsers.add_parser(
    "target-z", help="target z distribution", parents=[parser_hist]
)

parser_cloud_size = subparsers.add_parser(
    "cloud-size", help="point cloud size distribution", parents=[parser_hist]
)

args = parser.parse_args()

match args.subcommand:
    case "target-z":
        target_z(args.dataset, args.output, args.bins)
    case "cloud-size":
        cloud_size(args.dataset, args.output, args.bins)
