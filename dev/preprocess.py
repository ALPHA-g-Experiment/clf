#!/usr/bin/env python3

import argparse
import polars as pl
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Dataset preprocessing",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("dataset", help="path to the Parquet dataset")
parser.add_argument(
    "weights",
    type=int,
    nargs="+",
    help="integer weights for dataset splits (e.g. 1 1 for 50/50)",
)
parser.add_argument("--bins", type=int, default=100, help="number of flattening bins")
parser.add_argument("--output-dir", default=".", help="output directory")

args = parser.parse_args()

# Flatten the full dataset before splitting to avoid losing more data than
# necessary by e.g. fluctuation in binning in smaller subsets
lf = pl.scan_parquet(args.dataset).filter(pl.col("point_cloud").list.len() > 0)
z = lf.select(pl.col("target").arr.get(2)).collect()
breaks = pl.linear_space(
    z.min(), z.max(), num_samples=args.bins - 1, closed="none", eager=True
)
lf = lf.with_columns(bin=pl.col("target").arr.get(2).cut(breaks))

min_len = lf.group_by("bin").len().select(pl.col("len").min()).collect().item()
flat_lf = lf.filter(pl.int_range(pl.len()).shuffle(seed=0).over("bin") < min_len).drop(
    "bin"
)

total_counts = Counter(args.weights)
counter = Counter()

rows = flat_lf.select(pl.len()).collect().item()
offset = 0
for weight in args.weights:
    suffix = (
        str(weight)
        if total_counts[weight] == 1
        else f"{weight}{chr(97 + counter[weight])}"
    )
    output = Path(args.output_dir) / f"{Path(args.dataset).stem}_{suffix}.parquet"

    length = int(rows * weight / sum(args.weights))
    flat_lf.slice(offset, length).sink_parquet(output)
    print(f"Created `{output}`")

    counter[weight] += 1
    offset += length
