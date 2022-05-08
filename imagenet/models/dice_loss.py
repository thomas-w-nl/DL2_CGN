import torch

def DiceLoss(input, target):

    smoothing = 1

    input_flat = input.view(-1)
    target_flat = target.view(-1)

    numerator = 2 * (input_flat * target_flat) + smoothing

    denom = input_flat.sum() + target_flat.sum() + smoothing

    loss = numerator/denom

    return loss