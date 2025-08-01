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
        float: Average loss over the validation set.
    """
    model.eval()

    total_loss = 0.0
    metrics = {
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            total_loss += loss_fn(pred, y).item()

            signal_mask = y == 1.0
            background_mask = ~signal_mask
            # Raw logits because we are using BCE with logits
            positive_mask = pred >= 0.0
            negative_mask = ~positive_mask

            metrics["true_positives"] += (signal_mask & positive_mask).sum().item()
            metrics["true_negatives"] += (background_mask & negative_mask).sum().item()
            metrics["false_positives"] += (background_mask & positive_mask).sum().item()
            metrics["false_negatives"] += (signal_mask & negative_mask).sum().item()

    total_loss /= len(dataloader)
    accuracy = (metrics["true_positives"] + metrics["true_negatives"]) / sum(
        metrics.values()
    )

    return total_loss, accuracy
