import numpy as np
import torch


@torch.compile
def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    """
    Run one training epoch.

    Args:
        dataloader (DataLoader): PyTorch DataLoader yielding (input, target) batches.
        model (nn.Module): PyTorch model to train.
        loss_fn (callable): Loss function taking (pred, target) and returning a scalar loss Tensor.
        optimizer (Optimizer): PyTorch optimizer.
        device (torch.device): Device to perform computations on.

    Returns:
        float: Average loss over the epoch.
    """
    model.train()

    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def trim_sort(arr, trim_ratio):
    n = len(arr)
    sorted_arr = np.sort(arr)

    return sorted_arr[int(trim_ratio * n) : int((1 - trim_ratio) * n)]


@torch.compile
def test_one_epoch(dataloader, model, loss_fn, device):
    """
    Evaluate the model on a validation set.

    Args:
        dataloader (DataLoader): PyTorch DataLoader yielding (input, target) batches.
        model (nn.Module): PyTorch model to evaluate.
        loss_fn (callable): Loss function taking (pred, target) and returning a scalar loss Tensor.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: A tuple containing:
            - float: Average loss over the validation set.
            - float: Mean of (5% trimmed) residuals.
            - float: Standard deviation of (5% trimmed) residuals.
            - float: Sum of sliced absolute mean residuals.
    """
    model.eval()

    total_loss = 0.0
    residuals = []
    targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            total_loss += loss_fn(pred, y).item()
            residuals = np.append(residuals, (pred - y).cpu().numpy())
            targets = np.append(targets, y.cpu().numpy())
    total_loss /= len(dataloader)

    trimmed_residuals = trim_sort(residuals, 0.05)

    # Taking an up/down measurement in the bottom trap as a baseline:
    # Each of the up/down/lOc regions is about 200 mm.
    # The down region below mirror A extends down to -875 mm i.e. 275 mm from
    # the edge of the detector.
    # Then it makes sense to use 23 slices of ~100 mm each, ignoring the top and
    # bottom 2 slices.
    num_slices = 23
    slice_edges = np.linspace(-1152, 1152, num_slices, endpoint=False)[1:]
    indices = np.digitize(targets, slice_edges)

    sliced_abs_mean = 0.0
    for i in range(2, num_slices - 2):
        mask = indices == i
        if np.any(mask):
            trimmed = trim_sort(residuals[mask], 0.05)
            sliced_abs_mean += abs(trimmed.mean())

    return (
        total_loss,
        trimmed_residuals.mean(),
        trimmed_residuals.std(ddof=1),
        sliced_abs_mean,
    )
