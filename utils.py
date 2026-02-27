import torch


def compute_activation_sparsity(model):
    total = 0
    zeros = 0

    for module in model.modules():
        if hasattr(module, "weight"):
            w = module.weight.detach()
            total += w.numel()
            zeros += (w == 0).sum().item()

    if total == 0:
        return 0.0

    return zeros / total