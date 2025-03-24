import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from reco import pointnet_reg_mod_huber as model_module
from .SpacePointDataLoader import SpacePointLightDataset, SpacePointLightDataLoader
from reco import provider


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct the annihilation vertices for a single run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--data-path", type=str, default="./", help="path to data")
    parser.add_argument(
        "--data-prefix",
        type=str,
        default="spacepoints_vertices_simulation_311-341",
        help="data prefix",
    )
    parser.add_argument("--num-point", type=int, default=800, help="number of points")
    parser.add_argument("model", type=str, help="path to model")
    args = parser.parse_args()

    return args


def set_random_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_model(model, device, points, target, model_type):
    if model_type in [
        "shift and scale globally",
        "shift only",
        "scale and shift2",
        "shift only z",
    ]:
        points, mean_z = provider.normalize_spacepoints(
            points, False, event_type=model_type
        )
        normal_target = provider.normalize_spacepoints_target(
            target, event_type=model_type, mean_z=mean_z
        )
    else:
        points = provider.normalize_spacepoints(points, False)
        normal_target = provider.normalize_spacepoints_target(target)

    points = torch.Tensor(points).transpose(2, 1)
    normal_target = torch.Tensor(normal_target)
    target = torch.Tensor(target)

    pred_i, _ = model(points)

    if model_type in [
        "shift and scale globally",
        "shift only",
        "scale and shift2",
        "shift only z",
    ]:
        unnormal_pred_i = provider.unnormalize_spacepoints_target(
            pred_i.cpu(), event_type=model_type, mean_z=mean_z
        )
    else:
        unnormal_pred_i = provider.unnormalize_spacepoints_target(pred_i.cpu())

    return unnormal_pred_i, target


def main() -> None:
    args = parse_args()

    set_random_seeds(args.seed)

    test_dataset = SpacePointLightDataset(
        root=args.data_path,
        args=args,
        split="test",
    )
    testDataLoader = SpacePointLightDataLoader(test_dataset, batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model, map_location=device)

    regresser = model_module.get_model(normal_channel=False, use_wireamp=False)
    regresser.to(device)
    regresser.load_state_dict(state["model_state_dict"])
    regresser.eval()

    results_nn = []
    results_helix = []

    with torch.no_grad():
        for points, helix, target in tqdm(testDataLoader):
            points = points.data.numpy()
            target = target.data.numpy()
            target = np.reshape(target, (len(target), 1))
            points = points[(target != 0).flatten()]
            target = target[target != 0]
            target = np.reshape(target, (len(target), 1))
            helix = helix.data.numpy()
            helix = helix[(target != 0).flatten()]

            results_helix.extend((helix - target.flatten()))

            pred_i, targets = run_model(
                regresser, device, points, target, model_type="shift only z"
            )
            results_nn.extend((pred_i - targets).flatten())

    bins = np.linspace(-50, 50, 100)

    plt.hist(results_nn, bins, alpha=0.5, label="Model 1")
    plt.hist(results_helix, bins, alpha=0.5, label="Helix Fit")

    plt.legend(loc="upper right")
    plt.xlabel("Residuals")
    plt.ylabel("Counts")
    plt.show()


if __name__ == "__main__":
    main()
