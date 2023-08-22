import numpy as np
from scipy.optimize import minimize

import torch


__all__ = [
    "evaluate_ece",
    "evaluate_acc",
    "evaluate_nll",
    "evaluate_bs",
    "get_optimal_temperature",
]


@torch.no_grad()
def evaluate_ece(confidences: torch.Tensor, true_labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        true_labels (Tensor): a tensor of shape [N,] of ground truth labels.
        n_bins (int): the number of bins used by the histrogram binning.

    Returns:
        ece (float): expected calibration error of predictions.
    """
    # predicted labels and its confidences
    pred_confidences, pred_labels = torch.max(confidences, dim=1)

    # fixed binning (n_bins)
    ticks = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = ticks[:-1]
    bin_uppers = ticks[ 1:]

    # compute ECE across bins
    accuracies = pred_labels.eq(true_labels)
    ece = torch.zeros(1, device=confidences.device)
    avg_accuracies = []
    avg_confidences = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = pred_confidences.gt(bin_lower.item()) * pred_confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = pred_confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            avg_accuracies.append(accuracy_in_bin.item())
            avg_confidences.append(avg_confidence_in_bin.item())
        else:
            avg_accuracies.append(None)
            avg_confidences.append(None)

    return ece.item()


@torch.no_grad()
def evaluate_acc(confidences: torch.Tensor, true_labels: torch.Tensor) -> float:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        true_labels (Tensor): a tensor of shape [N,] of ground truth labels.

    Returns:
        acc (float): average accuracy of predictions.
    """

    acc = torch.max(confidences, dim=1)[1].eq(true_labels).float().mean().item()

    return acc


@torch.no_grad()
def evaluate_nll(confidences: torch.Tensor, true_labels: torch.Tensor) -> float:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        true_labels (Tensor): a tensor of shape [N,] of ground truth labels.

    Returns:
        nll (float): average negative-log-likelihood of predictions.
    """
    nll = torch.nn.functional.nll_loss(torch.log(1e-12 + confidences), true_labels).item()

    return nll


@torch.no_grad()
def evaluate_bs(confidences: torch.Tensor, true_labels: torch.Tensor) -> float:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        true_labels (Tensor): a tensor of shape [N,] of ground truth labels.

    Returns:
        bs (float): average Brier score of predictions.
    """

    targets = torch.eye(confidences.size(1), device=confidences.device)[true_labels].long()
    bs = torch.sum((confidences - targets)**2, dim=1).mean().item()

    return bs


@torch.no_grad()
def get_optimal_temperature(confidences: torch.Tensor, true_labels: torch.Tensor) -> float:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        true_labels (Tensor): a tensor of shape [N,] of ground truth labels.

    Returns:
        optimal_temperature (float): optimal value of temperature for given predictions.
    """

    def obj(t):
        target = true_labels.cpu().numpy()
        return -np.log(
            1e-12 + np.exp(
                torch.log_softmax(torch.log(1e-12 + confidences) / t, dim=1).data.numpy()
            )[np.arange(len(target)), target]
        ).mean()

    optimal_temperature = minimize(obj, 1.0, method="nelder-mead").x[0]

    return optimal_temperature
