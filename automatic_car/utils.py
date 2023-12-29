import torch
from torch.utils.data import DataLoader
from torch import nn


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    # Set the training mode for the model
    model.train()

    train_loss = 0

    for _, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    train_loss /= len(data_loader)

    print(f"Train loss: {train_loss}")
    return train_loss


def test(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: str = "cpu",
) -> float:
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
        # Test loss per batch
        test_loss /= len(data_loader)
        print(f"Test loss: {test_loss}")
        return test_loss
