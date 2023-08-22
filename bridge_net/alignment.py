from copy import deepcopy

import torch
import numpy as np
from scipy import optimize


def get_hook(name, act_store):
    def forward_hook(module, input, output):
        act_store[name] = output
    return forward_hook


def get_activations(model, x, block_idx):
    act_store = {}
    handles = []

    if block_idx < len(model.backbone.layers):
        block = model.backbone.layers[block_idx]
        if hasattr(block, "relu2"):  # BasicBlock, Bottleneck
            handles.append(block.relu1.register_forward_hook(get_hook("mid0", act_store)))
        if hasattr(block, "relu3"):  # Bottleneck
            handles.append(block.relu2.register_forward_hook(get_hook("mid1", act_store)))
        handles.append(block.register_forward_hook(get_hook("block", act_store)))
    else:
        handles.append(model.classifier.fc.register_forward_hook(get_hook("block", act_store)))

    model(x)

    for handle in handles:
        handle.remove()

    if "mid1" in act_store:
        return [act_store["mid0"], act_store["mid1"], act_store["block"]]
    elif "mid0" in act_store:
        return [act_store["mid0"], act_store["block"]]
    else:
        return [act_store["block"]]


def get_zscores(model, dataloader, block_idx):
    raw_act_lists = [[] for _ in range(3)]

    for image, _ in dataloader:
        image = image.to(model.device, non_blocking=False)

        raw_acts = get_activations(model, image, block_idx=block_idx)
        for i, raw_act in enumerate(raw_acts):
            if raw_act.dim() == 4:
                if raw_act.shape[-1] > 128:
                    raw_act_lists[i].append(raw_act[:, :, ::4, ::4].cpu())  # cut in quarter (memory saving)
                elif raw_act.shape[-1] > 32:
                    raw_act_lists[i].append(raw_act[:, :, ::2, ::2].cpu())  # cut in half (memory saving)
                else:
                    raw_act_lists[i].append(raw_act.cpu())
            else:
                raw_act_lists[i].append(raw_act.cpu())

    z_scores = []

    for raw_act_list in raw_act_lists:
        if len(raw_act_list) == 0:
            continue

        act = torch.cat([torch.transpose(v, 0, 1) for v in raw_act_list], dim=1)
        act = act.view(act.shape[0], -1)

        mu   = torch.mean(act, dim=1, keepdim=True)
        sigma = torch.std(act, dim=1, keepdim=True)

        z_score = (act - mu) / (sigma + 1e-5)
        z_scores.append(z_score)

    return z_scores  # [(mid0 z-score), (mid0 z-score), block z-score]


def compute_correlation(model0, model1, dataloader, block_idx):
    z_scores0 = get_zscores(model0, dataloader, block_idx)
    z_scores1 = get_zscores(model1, dataloader, block_idx)

    corrs = [
        torch.matmul(z_score0, z_score1.transpose(0, 1)) / z_score0.shape[1]
        for z_score0, z_score1 in zip(z_scores0, z_scores1)
    ]

    return corrs


@torch.inference_mode()
def compute_model_alignment(model0, model1, dataloader):
    num_layers = 1  # +1 for classifier
    sections = []

    for i, block in enumerate(model0.backbone.layers):
        block_name = block.__class__.__name__

        if block_name == "FirstBlock":
            num_layers += 1
        else:
            if block_name == "BasicBlock":
                num_layers += 2
            elif block_name == "Bottleneck":
                num_layers += 3
            else:
                raise ValueError("Unknown block type")

            if i == 1 or block.shortcut.__class__.__name__ == "ProjectionShortcut":
                sections.append(i)

    sections.append(len(model0.backbone.layers))

    align_idxs = []
    corr_means = []

    print("Block 0")

    *mid_corrs, block_corr = compute_correlation(model0, model1, dataloader, block_idx=0)
    align_idxs.append(compute_alignment(block_corr))
    corr_means.append(torch.mean(torch.diag(block_corr)))

    for block_idx, block in enumerate(model0.backbone.layers[1:], start=1):
        block_name = block.__class__.__name__

        if block_idx in sections:
            if block_name == "BasicBlock":
                num_filter = block.conv2.out_channels
            elif block_name == "Bottleneck":
                num_filter = block.conv3.out_channels
            else:
                raise ValueError("Unknown block type")

            update_list = []
            corr = torch.zeros([num_filter, num_filter])

        print(f"Block {block_idx}")

        corrs = compute_correlation(model0, model1, dataloader, block_idx=block_idx)  # [(mid0 corr), (mid1 corr), block corr]
        corr += corrs[-1]

        corr_means.extend([torch.mean(torch.diag(corr)) for corr in corrs])
        align_idxs.extend([compute_alignment(corr) for corr in corrs[:-1]] + [None])  # compute block corr later
        update_list.append(len(align_idxs) - 1)

        if block_idx + 1 in sections:
            block_align = compute_alignment(corr)
            for i in update_list:
                align_idxs[i] = block_align

    print(f"Classifier")

    *mid_corrs, block_corr = compute_correlation(model0, model1, dataloader, block_idx=len(model0.backbone.layers))
    corr_means.append(torch.mean(torch.diag(block_corr)))

    return align_idxs, corr_means


def compute_alignment(corr):
    align_idxs = optimize.linear_sum_assignment(1.01 - np.asarray(corr))[1]
    align_idxs = align_idxs.astype(int)
    return align_idxs


def align_weights_head(layer, match):
    layer_name = layer.__class__.__name__

    if layer_name in ["Conv2d", "Linear"]:
        assert len(match) == layer.weight.data.shape[0], "Permutation length does not match"
        layer.weight.data = layer.weight.data[match]
        if layer.bias is not None:
            layer.bias.data = layer.bias.data[match]
    elif layer_name == "FilterResponseNorm2d":
        assert len(match) == layer.gamma_frn.data.shape[1], "Permutation length does not match"
        layer.gamma_frn.data = layer.gamma_frn.data[:, match]
        layer.beta_frn.data = layer.beta_frn.data[:, match]
        layer.tau_frn.data = layer.tau_frn.data[:, match]
    elif layer_name == "BatchNorm2d":
        assert len(match) == layer.weight.data.shape[0], "Permutation length does not match"
        layer.weight.data = layer.weight.data[match]
        layer.bias.data = layer.bias.data[match]
        layer.running_mean = layer.running_mean[match]
        layer.running_var = layer.running_var[match]


def align_weights_tail(layer, match):
    layer_name = layer.__class__.__name__

    if layer_name in ["Conv2d", "Linear"]:
        assert len(match) == layer.weight.data.shape[1], "Permutation length does not match"
        layer.weight.data = layer.weight.data[:, match]


@torch.inference_mode()
def align_model(model, align_idxs):
    model = deepcopy(model)
    align_idxs = deepcopy(align_idxs)

    align_cur = 0

    for layer in model.backbone.layers[0].children():
        align_weights_head(layer, align_idxs[0])

    for block in model.backbone.layers[1:]:
        block_name = block.__class__.__name__

        if block_name == "BasicBlock":
            align_end = align_cur + 2
        elif block_name == "Bottleneck":
            align_end = align_cur + 3

        if block.shortcut.__class__.__name__ == "ProjectionShortcut":
            for layer in block.shortcut.children():
                align_weights_head(layer, align_idxs[align_end])
                align_weights_tail(layer, align_idxs[align_cur])
        else:
            # Alignment should be propogating through the res block
            align_idxs[align_end] = align_idxs[align_cur]

        for name, layer in block.named_children():
            if name.endswith("1") and not name.startswith("shortcut"):
                align_weights_head(layer, align_idxs[align_cur + 1])
                align_weights_tail(layer, align_idxs[align_cur])
            if name.endswith("2") and not name.startswith("shortcut"):
                align_weights_head(layer, align_idxs[align_cur + 2])
                align_weights_tail(layer, align_idxs[align_cur + 1])
            if name.endswith("3") and not name.startswith("shortcut"):
                align_weights_head(layer, align_idxs[align_cur + 3])
                align_weights_tail(layer, align_idxs[align_cur + 2])

        align_cur = align_end

    align_weights_tail(model.classifier.fc, align_idxs[-1])

    return model
