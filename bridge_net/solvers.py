import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from nxcl.config import ConfigDict


__all__ = [
    "SGD",
    "SWA",
    "build_sgd_optimizer",
    "build_warmup_simple_cosine_lr",
    "build_warmup_linear_decay_lr",
]


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        decoupled_weight_decay (bool, optional): enabled decoupled weight decay regularization (default: False)
    """

    def __init__(self, params, lr, momentum=0, weight_decay=0,
                 nesterov=False, decoupled_weight_decay=False) -> None:
        if lr < 0.0:
            raise ValueError("Invalid lr value: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, decoupled_weight_decay=decoupled_weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr                     = group['lr']
            momentum               = group['momentum']
            weight_decay           = group['weight_decay']
            nesterov               = group['nesterov']
            decoupled_weight_decay = group['decoupled_weight_decay']

            # target parameters which need to be updated
            params_with_grad     = []
            d_p_list             = []
            momentum_buffer_list = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            # update target parameters
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]

                if weight_decay != 0 and not decoupled_weight_decay:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p)
                    if nesterov:
                        d_p = d_p.add(buf)
                    else:
                        d_p = buf

                param.add_(d_p, alpha=-lr)

                if weight_decay != 0 and decoupled_weight_decay:
                    param.add_(weight_decay, alpha=-lr)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


class SWA(Optimizer):
    """Stochastic Weight Averaging (SWA)

    Args:
        base_optimizer (Optimizer):
            optimizer to use with SWA
    """
    def __init__(self, base_optimizer) -> None:
        self.base_optimizer = base_optimizer
        self.defaults       = self.base_optimizer.defaults
        self.param_groups   = self.base_optimizer.param_groups
        self.state          = self.base_optimizer.state
        for group in self.param_groups:
            group["n_avg"] = 0

    @torch.no_grad()
    def step(self, closure=None, sampling=False):
        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:

            for p in group["params"]:
                state = self.state[p]

                # save current parameters
                state["sgd_buffer"] = p.data

                # update SWA solution
                if sampling:
                    if "swa_buffer" not in state:
                        state["swa_buffer"] = torch.zeros_like(state["sgd_buffer"])
                    state["swa_buffer"].add_(
                        state["sgd_buffer"] - state["swa_buffer"],
                        alpha = 1.0 / float(group["n_avg"] + 1)
                    )

            if sampling:
                group["n_avg"] += 1

        return loss

    @torch.no_grad()
    def load_swa_buffer(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                p.data.copy_(state["swa_buffer"])

    @torch.no_grad()
    def load_sgd_buffer(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                p.data.copy_(state["sgd_buffer"])


def build_sgd_optimizer(model: nn.Module, **kwargs) -> Tuple[Optimizer, List]:

    # basic options
    BASE_LR                   = kwargs.pop("BASE_LR", None)
    WEIGHT_DECAY              = kwargs.pop("WEIGHT_DECAY", None)
    MOMENTUM                  = kwargs.pop("MOMENTUM", None)
    NESTEROV                  = kwargs.pop("NESTEROV", None)

    # options for BatchEnsemble
    SUFFIX_BE                 = kwargs.pop("SUFFIX_BE", tuple())
    BASE_LR_BE                = kwargs.pop("BASE_LR_BE", None)
    WEIGHT_DECAY_BE           = kwargs.pop("WEIGHT_DECAY_BE", None)
    MOMENTUM_BE               = kwargs.pop("MOMENTUM_BE", None)
    NESTEROV_BE               = kwargs.pop("NESTEROV_BE", None)

    _cache = set()
    params = list()
    for module in model.modules():

        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue

            if value in _cache:
                continue

            _cache.add(value)

            schedule_params = dict()
            schedule_params["params"] = [value]

            if module_param_name.endswith(tuple(SUFFIX_BE)):
                schedule_params["lr"]                     = BASE_LR_BE
                schedule_params["weight_decay"]           = WEIGHT_DECAY_BE
                schedule_params["momentum"]               = MOMENTUM_BE
                schedule_params["nesterov"]               = NESTEROV_BE
            else:
                schedule_params["lr"]                     = BASE_LR
                schedule_params["weight_decay"]           = WEIGHT_DECAY
                schedule_params["momentum"]               = MOMENTUM
                schedule_params["nesterov"]               = NESTEROV

            params.append(schedule_params)

    return SGD(params, lr=BASE_LR)


def build_warmup_simple_cosine_lr(optimizer: Optimizer, **kwargs) -> _LRScheduler:
    NUM_EPOCHS    = kwargs.pop("NUM_EPOCHS", None)
    WARMUP_EPOCHS = kwargs.pop("WARMUP_EPOCHS", None)
    WARMUP_METHOD = kwargs.pop("WARMUP_METHOD", None)
    WARMUP_FACTOR = kwargs.pop("WARMUP_FACTOR", None)

    def _lr_sched(epoch):
        # start from one
        epoch += 1

        # warmup
        if epoch < WARMUP_EPOCHS:
            if WARMUP_METHOD == "linear":
                return (1.0 - WARMUP_FACTOR) / (WARMUP_EPOCHS - 1) * (epoch - 1) + WARMUP_FACTOR
            elif WARMUP_METHOD == "constant":
                return WARMUP_FACTOR

        # cosine decays
        else:
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS + 1.0)))

    return LambdaLR(optimizer, lr_lambda=_lr_sched)


def build_warmup_linear_decay_lr(optimizer: Optimizer, **kwargs) -> _LRScheduler:
    MILESTONES    = kwargs.pop("MILESTONES", None)
    WARMUP_METHOD = kwargs.pop("WARMUP_METHOD", None)
    WARMUP_FACTOR = kwargs.pop("WARMUP_FACTOR", None)
    GAMMA         = kwargs.pop("GAMMA", None)

    def _lr_sched(epoch):

        # start from one
        epoch += 1

        # warmup
        if epoch < MILESTONES[0]:
            if WARMUP_METHOD == "linear":
                return (1.0 - WARMUP_FACTOR) / (MILESTONES[0] - 1) * (epoch - 1) + WARMUP_FACTOR
            elif WARMUP_METHOD == "constant":
                return WARMUP_FACTOR

        # high constant
        elif epoch < MILESTONES[1]:
            return 1.0

        # linear decay
        elif epoch < MILESTONES[2]:
            return (GAMMA - 1.0) / (MILESTONES[2] - MILESTONES[1]) * (epoch - MILESTONES[2]) + GAMMA

        # low constant
        else:
            return GAMMA

    return LambdaLR(optimizer, lr_lambda=_lr_sched)


def build_optimizer(cfg: ConfigDict, model: nn.Module) -> Optimizer:
    name = cfg.SOLVER.OPTIMIZER.NAME

    if name == "SGD":
        kwargs = dict()

        # basic options
        kwargs.update({
            "BASE_LR"               : cfg.SOLVER.OPTIMIZER.SGD.BASE_LR,
            "WEIGHT_DECAY"          : cfg.SOLVER.OPTIMIZER.SGD.WEIGHT_DECAY,
            "MOMENTUM"              : cfg.SOLVER.OPTIMIZER.SGD.MOMENTUM,
            "NESTEROV"              : cfg.SOLVER.OPTIMIZER.SGD.NESTEROV,
        })

        # options for BatchEnsemble
        if "BATCH_ENSEMBLE" in cfg.MODEL and cfg.MODEL.BATCH_ENSEMBLE.ENABLED:
            kwargs.update({
                "SUFFIX_BE"                 : cfg.SOLVER.OPTIMIZER.SGD.SUFFIX_BE,
                "BASE_LR_BE"                : cfg.SOLVER.OPTIMIZER.SGD.BASE_LR_BE,
                "WEIGHT_DECAY_BE"           : cfg.SOLVER.OPTIMIZER.SGD.WEIGHT_DECAY_BE,
                "MOMENTUM_BE"               : cfg.SOLVER.OPTIMIZER.SGD.MOMENTUM_BE,
                "NESTEROV_BE"               : cfg.SOLVER.OPTIMIZER.SGD.NESTEROV_BE,
            })

        optimizer = build_sgd_optimizer(model, **kwargs)

    else:
        raise NotImplementedError(f"Unknown cfg.SOLVER.OPTIMIZER.NAME = \"{name}\"")

    return optimizer


def build_scheduler(cfg: ConfigDict, optimizer: Optimizer) -> _LRScheduler:
    name = cfg.SOLVER.SCHEDULER.NAME

    if name == "WarmupSimpleCosineLR":
        kwargs = dict()
        kwargs.update({
            "NUM_EPOCHS"    : cfg.SOLVER.NUM_EPOCHS,
            "WARMUP_EPOCHS" : cfg.SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR.WARMUP_EPOCHS,
            "WARMUP_METHOD" : cfg.SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR.WARMUP_METHOD,
            "WARMUP_FACTOR" : cfg.SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR.WARMUP_FACTOR,
        })
        scheduler = build_warmup_simple_cosine_lr(optimizer, **kwargs)

    elif name == "WarmupLinearDecayLR":
        kwargs = dict()
        kwargs.update({
            "MILESTONES"    : cfg.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.MILESTONES,
            "WARMUP_METHOD" : cfg.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.WARMUP_METHOD,
            "WARMUP_FACTOR" : cfg.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.WARMUP_FACTOR,
            "GAMMA"         : cfg.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.GAMMA,
        })
        scheduler = build_warmup_linear_decay_lr(optimizer, **kwargs)

    else:
        raise NotImplementedError(f"Unknown cfg.SOLVER.SCHEDULER.NAME = \"{name}\"")

    return scheduler
