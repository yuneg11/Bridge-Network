import torch
from torch import nn



def is_bn(module):
    return module.__class__.__name__.startswith("BatchNorm")


def update_bn_stats(loader, model: nn.Module, device, **kwargs):
    model.train()

    momenta = {}

    def _reset_bn(module):
        if is_bn(module):
            module.reset_running_stats()

    def _get_momenta(module):
        if is_bn(module):
            momenta[module] = module.momentum

    def _set_momenta(module):
        if is_bn(module):
            module.momentum = momenta[module]

    model.apply(_reset_bn)
    model.apply(_get_momenta)

    num_samples = 0

    for images, _ in loader:
        images = images.to(device)
        batch_size = images.size(0)

        momentum = batch_size / (num_samples + batch_size)

        for module in momenta.keys():
            module.momentum = momentum

        model(images, **kwargs)
        num_samples += batch_size

    model.apply(_set_momenta)
