import torch


def DiceLoss(input, target):
    smoothing = 1

    input_flat = input.view(-1)
    target_flat = target.view(-1)

    numerator = 2 * (input_flat * target_flat).sum() + smoothing

    denom = input_flat.sum() + target_flat.sum() + smoothing

    loss = 1 - (numerator / denom)

    return loss
