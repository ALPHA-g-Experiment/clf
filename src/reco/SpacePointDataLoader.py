import os
import numpy as np
import h5py
import torch
import mmap
from torch.utils.data import Dataset


class SpacePointLightDataset(Dataset):
    def __init__(
        self,
        root,
        args,
        split="train",
        dataset_name="data",
        process_data="duplicate",
        sample_points="wireamp",
        return_runnumber=False,
        fformat="h5",
        just_zeros=False,
        xyz_target=False,
    ):
        self.root = root
        self.dataset_name = dataset_name
        self.process_data = process_data
        self.data_prefix = args.data_prefix

        self.length = None
        self.just_zeros = just_zeros

        self.npoints = args.num_point  # This might be useful

        self.use_wireamp = False
        self.sample_points = sample_points
        self.return_runnumber = return_runnumber
        self.use_simulation = True

        self.random_sample = True

        assert split in ["train", "test", "val_A", "val_B"]
        assert process_data == "duplicate" or process_data == "zeros"
        assert sample_points == "wireamp" or sample_points == "random"

        self.xyz_target = xyz_target

        if self.xyz_target:  # TEMPORARY
            self.file_path = os.path.join(
                root,
                f"spacepoints_vertices_simulation_0-310_0-148_{split}_light-wireamp0-runnum0-sim1-xyz1-flat.h5",
            )
        else:
            self.file_path = os.path.join(
                root,
                "%s_%s_light-wireamp%d-runnum%d-sim%d.%s"
                % (
                    self.data_prefix,
                    split,
                    int(self.use_wireamp),
                    int(self.return_runnumber),
                    int(self.use_simulation),
                    fformat,
                ),
            )

        with h5py.File(self.file_path, "r") as hf:
            assert self.dataset_name in hf, (
                f"{self.dataset_name} not found in {self.file_path}"
            )
            self.length = hf[self.dataset_name].shape[0]
            self.offset = hf[self.dataset_name].id.get_offset()  # added
            self.d_type = hf[self.dataset_name].dtype  # added
            self.shape = hf[self.dataset_name].shape  # added
            print(
                "reading:",
                os.path.join(
                    self.root,
                    "%s_%s_light-wireamp%d-runnum%d-sim%d.%s"
                    % (
                        self.data_prefix,
                        split,
                        int(self.use_wireamp),
                        int(self.return_runnumber),
                        int(self.use_simulation),
                        fformat,
                    ),
                ),
            )
            print(split, "dataset")
            print(self.length, "events stored")

        # Only have to memory map data --> new for afterrnoon 12-04-2023
        with open(self.file_path, "rb") as fh:
            fileno = fh.fileno()
            mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
            self.data_map = np.frombuffer(
                mapping,
                dtype=self.d_type,
                count=np.prod(self.shape),
                offset=self.offset,
            ).reshape(self.shape)

    def __len__(self):
        assert self.length is not None
        return self.length

    def pad(self, _x):
        if self.just_zeros:
            non_zero_mask = np.any(_x[:, :3] != 0, axis=1)
            _x = _x[non_zero_mask]

        else:
            r_cathode_min, r_cathode_max = 109.0, 190.0  # in mm
            radii = np.sqrt(_x[:, 0] ** 2 + _x[:, 1] ** 2)
            if sum(radii > r_cathode_max) > 0:
                print("r>190:", sum(radii >= r_cathode_max))
            valid_mask = (radii >= r_cathode_min) & (radii <= r_cathode_max)
            _x = _x[valid_mask]

        if len(_x) == 0:
            raise ValueError(
                "Filtered data is empty after removing points with r_cathode_min < r and r > r_cathode_max."
            )

        x = np.zeros((self.npoints, 3))

        if len(_x) < self.npoints:
            if self.process_data == "duplicate":
                x[: len(_x)] = _x[:, :3]
                dup_indices = np.arange(len(_x))
                while len(dup_indices) < self.npoints:
                    dup_indices = np.concatenate((dup_indices, np.arange(len(_x))))
                np.random.shuffle(dup_indices)
                x = _x[dup_indices[: self.npoints], :3]
            elif self.process_data == "zeros":
                x[: len(_x)] = _x[:, :3]
                random_indices = np.random.choice(
                    self.npoints, self.npoints, replace=False
                )
                x = x[random_indices]
        elif len(_x) == self.npoints:
            x = _x[:, :3]
        elif len(_x) > self.npoints:
            if self.sample_points == "wireamp":
                x = _x[: self.npoints, :3]
            elif self.sample_points == "random":
                random_indices = np.random.choice(len(_x), self.npoints, replace=False)
                x = _x[random_indices, :3]

        if self.xyz_target:
            vert = np.copy(_x[0, 6:9])
            simvert = np.copy(_x[0, 3:6])

        else:
            vert = np.copy(_x[0, 4])
            simvert = np.copy(_x[0, 3])

        x = torch.from_numpy(x)
        vert = torch.from_numpy(vert)
        simvert = torch.from_numpy(simvert)

        return x, vert, simvert

    def __getitem__(self, index):
        try:
            if self.random_sample:
                selected_indices = np.random.choice(
                    self.npoints, self.npoints, replace=False
                )
                _x = self.data_map[index][selected_indices]
                del selected_indices
            else:
                _x = self.data_map[index]

            x, vert, simvert = self.pad(_x)
            del _x

            return x, vert, simvert

        except ValueError as e:
            print(f"Skipping index {index} due to error: {e}")
            return None


def collate_fn(batch):
    # Filter out None values
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def SpacePointLightDataLoader(
    SpacePointLightDataset,
    batch_size=24,
    num_workers=4,
    sampler=None,
    pin_memory=None,
    shuffle=True,
):
    train_loader = torch.utils.data.DataLoader(
        SpacePointLightDataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=sampler,  # Use the custom collate function
        pin_memory=pin_memory,
    )
    return train_loader
