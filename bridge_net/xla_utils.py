import torch


try:
    import torch_xla
    is_xla = True
except ImportError:
    is_xla = False

if is_xla:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    from nxml.dev.torch.xla import dist
    is_xla = True

    NUM_AVAILABLE_DEVICES = 8   # For Cloud TPUs
    optimizer_step = lambda optimizer, **kwargs: xm.optimizer_step(optimizer, optimizer_args=kwargs)
    create_ddp_model = lambda model: model
    synchronize = lambda name: xm.rendezvous(name)

    def get_state_dict(model):
        target = model

        if isinstance(target, torch.nn.parallel.DistributedDataParallel):
            target = target.module
        if isinstance(model, torch.optim.swa_utils.AveragedModel):
            target = target.module

        state_dict = target.state_dict()

        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].cpu()

        return state_dict

else:
    from nxml.dev.torch import dist
    is_xla = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    NUM_AVAILABLE_DEVICES = torch.cuda.device_count()
    optimizer_step = lambda optimizer, **kwargs: optimizer.step(**kwargs)
    create_ddp_model = lambda model: dist.create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=False)
    synchronize = lambda name: dist.synchronize()

    def get_state_dict(model):
        target = model

        if isinstance(target, torch.nn.parallel.DistributedDataParallel):
            target = target.module
        if isinstance(model, torch.optim.swa_utils.AveragedModel):
            target = target.module

        return target.state_dict()
