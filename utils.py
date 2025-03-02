import torch

def accuracy(out: torch.Tensor, y: torch.Tensor) -> float:
    """Computes the accuracy of the predictions."""
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float().item()

def get_weights(net: torch.nn.Module) -> torch.tensor:
    """Extracts the weights of a neural network into a single tensor."""
    return torch.cat([p.view(-1).detach().cpu() for p in net.parameters()])