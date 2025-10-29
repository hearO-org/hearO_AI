import torch
def accuracy(pred, target):
    with torch.no_grad():
        p = pred.argmax(1)
        return (p == target).float().mean().item()
