import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "ReLU",
    "SiLU",
    "Identity",
    "MaxPool2d",
    "Conv2d",
    "Conv2d_Bezier",
    "Conv2d_BatchEnsemble",
    "Linear",
    "Linear_Bezier",
    "Linear_BatchEnsemble",
    "FilterResponseNorm2d",
    "FilterResponseNorm2d_Bezier",
    "BatchNorm2d",
    "BatchNorm2d_Bezier",
]


def initialize_tensor(
    tensor: torch.Tensor,
    initializer: str,
    init_values: List[float] = [],
) -> None:

    if initializer == "zeros":
        nn.init.zeros_(tensor)
    elif initializer == "ones":
        nn.init.ones_(tensor)
    elif initializer == "uniform":
        nn.init.uniform_(tensor, init_values[0], init_values[1])
    elif initializer == "normal":
        nn.init.normal_(tensor, init_values[0], init_values[1])
    elif initializer == "random_sign":
        with torch.no_grad():
            tensor.data.copy_(
                2.0 * init_values[1] * torch.bernoulli(
                    torch.zeros_like(tensor) + init_values[0]
                ) - init_values[1]
            )
    else:
        raise NotImplementedError(f"Unknown initializer: {initializer}")


class ReLU(nn.ReLU):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class SiLU(nn.SiLU):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class Identity(nn.Identity):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class MaxPool2d(nn.MaxPool2d):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.same_padding = kwargs.pop("same_padding", False)
        if self.same_padding:
            kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.same_padding:
            x = self._pad_input(x)
        return self._conv_forward(x, self.weight, self.bias)


class Conv2d_Bezier(Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.ParameterList([self._parameters.pop("weight", None)])
        if self.bias is not None:
            self.bias = nn.ParameterList([self._parameters.pop("bias", None)])

    @torch.no_grad()
    def add_param(self) -> None:
        _p = nn.Parameter(self.weight[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.weight) / len(self.weight))
        self.weight.append(_p)
        if self.bias is not None:
            _p = nn.Parameter(self.bias[-1].detach().clone())
            _p.data.copy_(torch.zeros_like(_p) + sum(self.bias) / len(self.bias))
            self.bias.append(_p)

    def freeze_param(self, index: int) -> None:
        self.weight[index].grad = None
        self.weight[index].requires_grad = False
        if self.bias is not None:
            self.bias[index].grad = None
            self.bias[index].requires_grad = False

    def _sample_parameters(self, λ: float) -> Tuple[torch.Tensor]:
        w = torch.zeros_like(self.weight[0])
        b = torch.zeros_like(self.bias[0]) if self.bias is not None else None

        if len(self.weight) == 1:
            w += self.weight[0]
            if b is not None:
                b += self.bias[0]

        elif len(self.weight) == 2:
            w += (1 - λ) * self.weight[0]
            w += λ * self.weight[1]
            if b is not None:
                b += (1 - λ) * self.bias[0]
                b += λ * self.bias[1]

        elif len(self.weight) == 3:
            w += (1 - λ) * (1 - λ) * self.weight[0]
            w += 2 * (1 - λ) * λ * self.weight[1]
            w += λ * λ * self.weight[2]
            if b is not None:
                b += (1 - λ) * (1 - λ) * self.bias[0]
                b += 2 * (1 - λ) * λ * self.bias[1]
                b += λ * λ * self.bias[2]

        else:
            raise NotImplementedError()

        return w, b

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        kh, kw = self.weight[0].size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x

    def forward(self, x: torch.Tensor, bezier_lambda: float = 0., **kwargs) -> torch.Tensor:
        weight, bias = self._sample_parameters(bezier_lambda)
        if self.same_padding:
            x = self._pad_input(x)
        return self._conv_forward(x, weight, bias)


class Conv2d_BatchEnsemble(Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        ensemble_size     = kwargs.pop("ensemble_size", None)
        alpha_initializer = kwargs.pop("alpha_initializer", None)
        gamma_initializer = kwargs.pop("gamma_initializer", None)
        use_ensemble_bias = kwargs.pop("use_ensemble_bias", None)
        super(Conv2d_BatchEnsemble, self).__init__(*args, **kwargs)

        self.ensemble_size     = ensemble_size
        self.alpha_initializer = alpha_initializer
        self.gamma_initializer = gamma_initializer

        # register parameters
        self.register_parameter("alpha_be", nn.Parameter(torch.Tensor(self.ensemble_size, self.in_channels)))
        self.register_parameter("gamma_be", nn.Parameter(torch.Tensor(self.ensemble_size, self.out_channels)))
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter("ensemble_bias", nn.Parameter(torch.Tensor(self.ensemble_size, self.out_channels)))
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        initialize_tensor(self.alpha_be, **self.alpha_initializer)
        initialize_tensor(self.gamma_be, **self.gamma_initializer)
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        _, C1, H1, W1 = x.size()
        r_x = x.view(self.ensemble_size, -1, C1, H1, W1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, C1, 1, 1)
        r_x = r_x.view(-1, C1, H1, W1)

        if self.same_padding:
            r_x = self._pad_input(r_x)
        w_r_x = self._conv_forward(r_x, self.weight, self.bias)

        _, C2, H2, W2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, C2, H2, W2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, C2, 1, 1)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, C2, 1, 1)
        s_w_r_x = s_w_r_x.view(-1, C2, H2, W2)

        return s_w_r_x

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, ensemble_size={ensemble_size}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            if self.ensemble_bias is None:
                s += ', bias=False, ensemble_bias=False'
            else:
                s += ', bias=False, ensemble_bias=True'
        else:
            if self.ensemble_bias is None:
                s += ', bias=True, ensemble_bias=False'
            else:
                s += ', bias=True, ensemble_bias=True'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class Linear(nn.Linear):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(x)


class Linear_Bezier(Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = nn.ParameterList([self._parameters.pop("weight", None)])
        if self.bias is not None:
            self.bias = nn.ParameterList([self._parameters.pop("bias", None)])

    @torch.no_grad()
    def add_param(self) -> None:
        _p = nn.Parameter(self.weight[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.weight) / len(self.weight))
        self.weight.append(_p)
        if self.bias is not None:
            _p = nn.Parameter(self.bias[-1].detach().clone())
            _p.data.copy_(torch.zeros_like(_p) + sum(self.bias) / len(self.bias))
            self.bias.append(_p)

    def freeze_param(self, index: int) -> None:
        self.weight[index].grad = None
        self.weight[index].requires_grad = False
        if self.bias is not None:
            self.bias[index].grad = None
            self.bias[index].requires_grad = False

    def _sample_parameters(self, λ: float) -> Tuple[torch.Tensor]:
        w = torch.zeros_like(self.weight[0])
        b = torch.zeros_like(self.bias[0]) if self.bias is not None else None

        if len(self.weight) == 1:
            w += self.weight[0]
            if b is not None:
                b += self.bias[0]

        elif len(self.weight) == 2:
            w += (1 - λ) * self.weight[0]
            w += λ * self.weight[1]
            if b is not None:
                b += (1 - λ) * self.bias[0]
                b += λ * self.bias[1]

        elif len(self.weight) == 3:
            w += (1 - λ) * (1 - λ) * self.weight[0]
            w += 2 * (1 - λ) * λ * self.weight[1]
            w += λ * λ * self.weight[2]
            if b is not None:
                b += (1 - λ) * (1 - λ) * self.bias[0]
                b += 2 * (1 - λ) * λ * self.bias[1]
                b += λ * λ * self.bias[2]

        else:
            raise NotImplementedError()

        return w, b

    def forward(self, x: torch.Tensor, bezier_lambda: float = 0., **kwargs) -> torch.Tensor:
        weight, bias = self._sample_parameters(bezier_lambda)
        return F.linear(x, weight, bias)


class Linear_BatchEnsemble(Linear):
    def __init__(self, *args, **kwargs) -> None:
        ensemble_size     = kwargs.pop("ensemble_size", None)
        alpha_initializer = kwargs.pop("alpha_initializer", None)
        gamma_initializer = kwargs.pop("gamma_initializer", None)
        use_ensemble_bias = kwargs.pop("use_ensemble_bias", None)
        super(Linear_BatchEnsemble, self).__init__(*args, **kwargs)

        self.ensemble_size     = ensemble_size
        self.alpha_initializer = alpha_initializer
        self.gamma_initializer = gamma_initializer

        # register parameters
        self.register_parameter("alpha_be", nn.Parameter(torch.Tensor(self.ensemble_size, self.in_features)))
        self.register_parameter("gamma_be", nn.Parameter(torch.Tensor(self.ensemble_size, self.out_features)))
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter("ensemble_bias", nn.Parameter(torch.Tensor(self.ensemble_size, self.out_features)))
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        initialize_tensor(self.alpha_be, **self.alpha_initializer)
        initialize_tensor(self.gamma_be, **self.gamma_initializer)
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        _, D1 = x.size()
        r_x = x.view(self.ensemble_size, -1, D1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, D1)
        r_x = r_x.view(-1, D1)

        w_r_x = nn.functional.linear(r_x, self.weight, self.bias)

        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, D2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, D2)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, D2)
        s_w_r_x = s_w_r_x.view(-1, D2)

        return s_w_r_x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, ensemble_size={}, ensemble_bias={}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.ensemble_size, self.ensemble_bias is not None
        )


class FilterResponseNorm2d(nn.Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-6,
            learnable_eps: bool = False,
            learnable_eps_init: float = 1e-4,
        ) -> None:
        super(FilterResponseNorm2d, self).__init__()
        self.num_features       = num_features
        self.eps                = eps
        self.learnable_eps      = learnable_eps
        self.learnable_eps_init = learnable_eps_init

        self.gamma_frn = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta_frn  = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.tau_frn   = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        if self.learnable_eps:
            self.eps_l_frn = nn.Parameter(torch.Tensor(1))
        else:
            self.register_buffer(
                name="eps_l_frn",
                tensor=torch.zeros(1),
                persistent=False
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.gamma_frn)
        nn.init.zeros_(self.beta_frn)
        nn.init.zeros_(self.tau_frn)
        if self.learnable_eps:
            nn.init.constant_(self.eps_l_frn, self.learnable_eps_init)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def extra_repr(self):
        return '{num_features}, eps={eps}, learnable_eps={learnable_eps}'.format(**self.__dict__)

    def _norm_forward(
            self,
            x: torch.Tensor,
            γ: torch.Tensor,
            β: torch.Tensor,
            τ: torch.Tensor,
            ε: torch.Tensor,
        ) -> torch.Tensor:
        ν2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(ν2 + ε)
        x = γ * x + β
        x = torch.max(x, τ)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self._check_input_dim(x)
        return self._norm_forward(x, self.gamma_frn, self.beta_frn,
                                  self.tau_frn, self.eps + self.eps_l_frn.abs())


class FilterResponseNorm2d_Bezier(FilterResponseNorm2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gamma_frn = nn.ParameterList([self._parameters.pop("gamma_frn", None)])
        self.beta_frn = nn.ParameterList([self._parameters.pop("beta_frn", None)])
        self.tau_frn = nn.ParameterList([self._parameters.pop("tau_frn", None)])
        if "eps_l_frn" in self._parameters:
            self.eps_l_frn = nn.ParameterList([self._parameters.pop("eps_l_frn", None)])

    @torch.no_grad()
    def add_param(self) -> None:
        _p = nn.Parameter(self.gamma_frn[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.gamma_frn) / len(self.gamma_frn))
        self.gamma_frn.append(_p)
        _p = nn.Parameter(self.beta_frn[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.beta_frn) / len(self.beta_frn))
        self.beta_frn.append(_p)
        _p = nn.Parameter(self.tau_frn[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.tau_frn) / len(self.tau_frn))
        self.tau_frn.append(_p)
        if isinstance(self.eps_l_frn, nn.ParameterList):
            _p = nn.Parameter(self.eps_l_frn[-1].detach().clone())
            _p.data.copy_(torch.zeros_like(_p) + sum(self.eps_l_frn) / len(self.eps_l_frn))
            self.eps_l_frn.append(_p)

    def freeze_param(self, index: int) -> None:
        self.gamma_frn[index].grad = None
        self.gamma_frn[index].requires_grad = False
        self.beta_frn[index].grad = None
        self.beta_frn[index].requires_grad = False
        self.tau_frn[index].grad = None
        self.tau_frn[index].requires_grad = False
        if isinstance(self.eps_l_frn, nn.ParameterList):
            self.eps_l_frn[index].grad = None
            self.eps_l_frn[index].requires_grad = False

    def _sample_parameters(self, λ: float) -> Tuple[torch.Tensor]:
        g = torch.zeros_like(self.gamma_frn[0])
        b = torch.zeros_like(self.beta_frn[0])
        t = torch.zeros_like(self.tau_frn[0])
        e = torch.zeros_like(self.eps_l_frn[0]) if isinstance(self.eps_l_frn, nn.ParameterList) else self.eps_l_frn

        if len(self.gamma_frn) == 1:
            g += self.gamma_frn[0]
            b += self.beta_frn[0]
            t += self.tau_frn[0]
            if isinstance(self.eps_l_frn, nn.ParameterList):
                e += self.eps_l_frn[0]

        elif len(self.gamma_frn) == 2:
            g += (1 - λ) * self.gamma_frn[0]
            g += λ * self.gamma_frn[1]
            b += (1 - λ) * self.beta_frn[0]
            b += λ * self.beta_frn[1]
            t += (1 - λ) * self.tau_frn[0]
            t += λ * self.tau_frn[1]
            if isinstance(self.eps_l_frn, nn.ParameterList):
                e += (1 - λ) * self.eps_l_frn[0]
                e += λ * self.eps_l_frn[1]

        elif len(self.gamma_frn) == 3:
            g += (1 - λ) * (1 - λ) * self.gamma_frn[0]
            g += 2 * (1 - λ) * λ * self.gamma_frn[1]
            g += λ * λ * self.gamma_frn[2]
            b += (1 - λ) * (1 - λ) * self.beta_frn[0]
            b += 2 * (1 - λ) * λ * self.beta_frn[1]
            b += λ * λ * self.beta_frn[2]
            t += (1 - λ) * (1 - λ) * self.tau_frn[0]
            t += 2 * (1 - λ) * λ * self.tau_frn[1]
            t += λ * λ * self.tau_frn[2]
            if isinstance(self.eps_l_frn, nn.ParameterList):
                e += (1 - λ) * (1 - λ) * self.eps_l_frn[0]
                e += 2 * (1 - λ) * λ * self.eps_l_frn[1]
                e += λ * λ * self.eps_l_frn[2]

        else:
            raise NotImplementedError()

        e = e.abs() + self.eps

        return g, b, t, e

    def forward(self, x: torch.Tensor, bezier_lambda: float = 0., **kwargs) -> torch.Tensor:
        self._check_input_dim(x)
        g, b, t, e = self._sample_parameters(bezier_lambda)
        return self._norm_forward(x, g, b, t, e)


class BatchNorm2d(nn.BatchNorm2d):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class BatchNorm2d_Bezier(BatchNorm2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = nn.ParameterList([self._parameters.pop("weight", None)])
        self.bias   = nn.ParameterList([self._parameters.pop("bias",   None)])

    @torch.no_grad()
    def add_param(self) -> None:
        _p = nn.Parameter(self.weight[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.weight) / len(self.weight))
        self.weight.append(_p)
        _p = nn.Parameter(self.bias[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.bias) / len(self.bias))
        self.bias.append(_p)

    def freeze_param(self, index: int) -> None:
        self.weight[index].grad = None
        self.weight[index].requires_grad = False
        self.bias[index].grad = None
        self.bias[index].requires_grad = False

    def _sample_parameters(self, λ: float) -> Tuple[torch.Tensor]:
        w = torch.zeros_like(self.weight[0])
        b = torch.zeros_like(self.bias[0])

        if len(self.weight) == 1:
            w += self.weight[0]
            b += self.bias[0]

        elif len(self.weight) == 2:
            w += (1 - λ) * self.weight[0]
            w += λ * self.weight[1]
            b += (1 - λ) * self.bias[0]
            b += λ * self.bias[1]

        elif len(self.weight) == 3:
            w += (1 - λ) * (1 - λ) * self.weight[0]
            w += 2 * (1 - λ) * λ * self.weight[1]
            w += λ * λ * self.weight[2]
            b += (1 - λ) * (1 - λ) * self.bias[0]
            b += 2 * (1 - λ) * λ * self.bias[1]
            b += λ * λ * self.bias[2]

        else:
            raise NotImplementedError()

        return w, b

    def forward(self, input: torch.Tensor, bezier_lambda: float = 0., **kwargs) -> torch.Tensor:
        self._check_input_dim(input)
        w, b = self._sample_parameters(bezier_lambda)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return nn.functional.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            w, # self.weight,
            b, # self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
