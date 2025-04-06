import polars as pl
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, parquet_file, cloud_size):
        self.inner = (
            pl.scan_parquet(parquet_file)
            .with_columns(
                pl.col("target").arr.get(2),
                pl.col("point_cloud")
                .list.eval(
                    pl.element().sort_by(
                        pl.element().arr.get(0).pow(2) + pl.element().arr.get(1).pow(2)
                    )
                )
                .list.gather(
                    pl.int_range(cloud_size),
                    null_on_oob=True,
                )
                .list.eval(pl.element().fill_null(strategy="forward"))
                .list.to_array(cloud_size),
            )
            .collect()
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        # If `cloud_size * 3 * dataset_rows > 2^32`, calling `to_torch` on the
        # complete dataset panics.
        return self.inner[idx].to_torch("dataset", label="target")[0]
