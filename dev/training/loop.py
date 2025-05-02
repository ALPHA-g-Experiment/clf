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
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            total_loss += loss_fn(pred, y).item()

    return total_loss / len(dataloader)
