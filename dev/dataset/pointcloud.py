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
                    pl.int_ranges(cloud_size) % pl.col("point_cloud").list.len()
                )
                .list.to_array(cloud_size),
            )
            .collect()
            .to_torch("dataset", label="target")
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self.inner[idx]
