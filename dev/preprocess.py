#!/usr/bin/env python3

import argparse
import polars as pl
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Dataset preprocessing",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("pbar_dataset", help="path to the signal Parquet dataset")
parser.add_argument("cosmic_dataset", help="path to the background Parquet dataset")
parser.add_argument(
    "weights",
    type=int,
    nargs="+",
    help="integer weights for dataset splits (e.g. 1 1 for 50/50)",
)
parser.add_argument("--output-dir", default=".", help="output directory")

args = parser.parse_args()

signal_df = (
    pl.scan_parquet(args.pbar_dataset)
    .with_columns(target=pl.lit(1.0).cast(pl.Float32))
    .collect()
)
background_df = (
    pl.scan_parquet(args.cosmic_dataset)
    .with_columns(target=pl.lit(0.0).cast(pl.Float32))
    .collect()
)

n_rows = min(signal_df.height, background_df.height)
signal_df = signal_df.sample(n_rows, shuffle=True, seed=0)
background_df = background_df.sample(n_rows, shuffle=True, seed=0)

total_counts = Counter(args.weights)
counter = Counter()

offset = 0
for weight in args.weights:
    suffix = (
        str(weight)
        if total_counts[weight] == 1
        else f"{weight}{chr(97 + counter[weight])}"
    )
    output = (
        Path(args.output_dir)
        / f"{Path(args.pbar_dataset).stem}_{Path(args.cosmic_dataset).stem}_{suffix}.parquet"
    )

    length = int(n_rows * weight / sum(args.weights))
    pl.concat(
        [signal_df.slice(offset, length), background_df.slice(offset, length)]
    ).sample(fraction=1.0, shuffle=True, seed=0).write_parquet(output)
    print(f"Created `{output}`")

    counter[weight] += 1
    offset += length
