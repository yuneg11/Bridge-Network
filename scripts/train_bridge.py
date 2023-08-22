import os
import sys

sys.path.insert(0, os.getcwd())

import logging
import argparse

import torch

from tabulate import tabulate

import nxcl
from nxcl.rich import Progress
from nxcl.config import load_config, save_config, add_config_arguments, ConfigDict
from nxcl.dev import utils as dev_utils

import nxml
from nxml.torch.nn import functional as F

from bridge_net.xla_utils import *
from bridge_net.data import build_dataloaders
from bridge_net.modeling import build_model, build_bridge
from bridge_net.solvers import build_optimizer, build_scheduler


@torch.no_grad()
def mixup(images, distribution=None):
    if distribution is None:
        return images

    lam = distribution.sample((images.shape[0],)).to(images.device)
    lam = lam.view(images.shape[0], 1, 1, 1)
    index = torch.randperm(images.shape[0], device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed


def train_step(ctx, batch, bezier_model, model, optimizer):
    cfg: ConfigDict = ctx["config"]
    images, labels = batch

    images = mixup(images, ctx["distribution"])

    with torch.no_grad():
        bezier_outputs_0 = bezier_model(images, bezier_lambda=0.0)
        bezier_outputs_c = bezier_model(images, bezier_lambda=0.5)
        if cfg.MODEL.TYPE == 2:
            bezier_outputs_1 = bezier_model(images, bezier_lambda=1.0)

    if cfg.MODEL.TYPE == 1:
        outputs = model(torch.cat([bezier_outputs_0[r] for r in cfg.MODEL.REQUIRES], dim=1))
    elif cfg.MODEL.TYPE == 2:
        outputs = model(
            torch.cat([bezier_outputs_0[r] for r in cfg.MODEL.REQUIRES], dim=1),
            torch.cat([bezier_outputs_1[r] for r in cfg.MODEL.REQUIRES], dim=1),
        )

    pred     = outputs[cfg.SOLVER.MATCH]
    target_0 = bezier_outputs_0[cfg.SOLVER.MATCH]
    target_c = bezier_outputs_c[cfg.SOLVER.MATCH]
    if cfg.MODEL.TYPE == 2:
        target_1 = bezier_outputs_1[cfg.SOLVER.MATCH]

    if cfg.SOLVER.MATCH == "logits":
        loss_orig = F.mse_loss(input=pred, target=target_c)

        if cfg.SOLVER.REGULARIZATION > 0.:
            if cfg.MODEL.TYPE == 1:
                reg = cfg.SOLVER.REGULARIZATION * F.mse_loss(input=pred, target=target_0)
            elif cfg.MODEL.TYPE == 2:
                reg = cfg.SOLVER.REGULARIZATION * (
                    F.mse_loss(input=pred, target=target_0) + F.mse_loss(input=pred, target=target_1)
                ) / 2.
        else:
            reg = 0.

    elif cfg.SOLVER.MATCH == "log_confidences":
        loss_orig = F.kl_div(input=pred, target=target_c, log_target=True, reduction="batchmean")

        if cfg.SOLVER.REGULARIZATION > 0.:
            if cfg.MODEL.TYPE == 1:
                reg = cfg.SOLVER.REGULARIZATION * F.kl_div(input=pred, target=target_0, log_target=True, reduction="batchmean")
            elif cfg.MODEL.TYPE == 2:
                reg = cfg.SOLVER.REGULARIZATION * (
                    F.kl_div(input=pred, target=target_0, log_target=True, reduction="batchmean") +
                    F.kl_div(input=pred, target=target_1, log_target=True, reduction="batchmean")
                ) / 2.
        else:
            reg = 0.

    loss = loss_orig - reg

    optimizer.zero_grad()
    loss.backward()
    optimizer_step(optimizer)

    with torch.inference_mode():
        # NOTE: This is not synchronized metrics, but it is ok for training
        bezier_labels = target_c.argmax(dim=-1)
        acc1_true   = F.accuracy(input=pred, target=labels, topk=1)
        acc1_bezier = F.accuracy(input=pred, target=bezier_labels, topk=1)

    return {"loss": loss.detach(), "acc1_true": acc1_true, "acc1_bezier": acc1_bezier, "loss_orig": loss_orig, "reg": reg}


def valid_step(ctx, batch, bezier_model, model):
    cfg: ConfigDict = ctx["config"]

    images, labels = batch

    bezier_outputs_0 = bezier_model(images, bezier_lambda=0.0)
    bezier_outputs_c = bezier_model(images, bezier_lambda=0.5)
    if cfg.MODEL.TYPE == 2:
        bezier_outputs_1 = bezier_model(images, bezier_lambda=1.0)

    if cfg.MODEL.TYPE == 1:
        outputs = model(torch.cat([bezier_outputs_0[r] for r in cfg.MODEL.REQUIRES], dim=1))
    elif cfg.MODEL.TYPE == 2:
        outputs = model(
            torch.cat([bezier_outputs_0[r] for r in cfg.MODEL.REQUIRES], dim=1),
            torch.cat([bezier_outputs_1[r] for r in cfg.MODEL.REQUIRES], dim=1),
        )

    pred     = outputs[cfg.SOLVER.MATCH]
    target_0 = bezier_outputs_0[cfg.SOLVER.MATCH]
    target_c = bezier_outputs_c[cfg.SOLVER.MATCH]
    if cfg.MODEL.TYPE == 2:
        target_1 = bezier_outputs_1[cfg.SOLVER.MATCH]

    if cfg.SOLVER.MATCH == "logits":
        loss_orig = F.mse_loss(input=pred, target=target_c)

        if cfg.SOLVER.REGULARIZATION > 0.:
            if cfg.MODEL.TYPE == 1:
                reg = cfg.SOLVER.REGULARIZATION * F.mse_loss(input=pred, target=target_0)
            elif cfg.MODEL.TYPE == 2:
                reg = cfg.SOLVER.REGULARIZATION * (
                    F.mse_loss(input=pred, target=target_0) + F.mse_loss(input=pred, target=target_1)
                )
        else:
            reg = 0.

    elif cfg.SOLVER.MATCH == "log_confidences":
        loss_orig = F.kl_div(input=pred, target=target_c, log_target=True, reduction="batchmean")

        if cfg.SOLVER.REGULARIZATION > 0.:
            if cfg.MODEL.TYPE == 1:
                reg = cfg.SOLVER.REGULARIZATION * F.kl_div(input=pred, target=target_0, log_target=True, reduction="batchmean")
            elif cfg.MODEL.TYPE == 2:
                reg = cfg.SOLVER.REGULARIZATION * (
                    F.kl_div(input=pred, target=target_0, log_target=True, reduction="batchmean") +
                    F.kl_div(input=pred, target=target_1, log_target=True, reduction="batchmean")
                )
        else:
            reg = 0.

    loss = loss_orig - reg

    # NOTE: Simply, each device iterates over the whole dataset.
    #       It is not efficient, but currently distributed communication is more expensive.
    bezier_labels = target_c.argmax(dim=-1)
    acc1_true   = F.accuracy(input=pred, target=labels, topk=1)
    acc1_bezier = F.accuracy(input=pred, target=bezier_labels, topk=1)

    return {"loss": loss, "acc1_true": acc1_true, "acc1_bezier": acc1_bezier, "loss_orig": loss_orig, "reg": reg}


def train_epoch(ctx, bezier_model, model, optimizer, scheduler, train_dataloader, device):
    model.train()

    if ctx["is_distributed"]:
        (train_dataloader._loader if is_xla else train_dataloader).sampler.set_epoch(ctx["epoch_idx"])

    meter = dev_utils.AverageMeter("loss", "acc1_true", "acc1_bezier", "loss_orig", "reg")

    p, logging_batches, nb_fmt, cfg = (ctx["progress"], ctx["logging_batches"], ctx["nb_fmt"], ctx["config"])
    num_batch = len(train_dataloader)

    for batch_idx, (images, labels) in enumerate(p.track(train_dataloader, description="Train", remove=True), start=1):
        batch = (images.to(device), labels.to(device))
        metrics = train_step(ctx, batch=batch, bezier_model=bezier_model, model=model, optimizer=optimizer)
        meter.update(metrics, n=len(batch[0]))

        if batch_idx in logging_batches:
            logging.info(
                ctx["epoch_header"]
                + f"[Batch {batch_idx:{nb_fmt}}/{num_batch:{nb_fmt}}] "
                + f"Loss {meter.loss:8.5f}  "
                + (f"Loss/-reg {meter.loss_orig:.5f}  " if cfg.SOLVER.REGULARIZATION > 0. else "")
                + (f"Reg {meter.reg:.5f}  " if cfg.SOLVER.REGULARIZATION > 0. else "")
                + f"Acc/True {meter.acc1_true * 100:6.2f}  "
                + f"Acc/Bezier {meter.acc1_bezier * 100:6.2f}  "
                + f"LR {scheduler.get_last_lr()[0]:.3e}  "
            )

    scheduler.step()

    return meter


@torch.no_grad()
def valid_epoch(ctx, bezier_model, model, valid_dataloader, device):
    model.eval()
    meter = dev_utils.AverageMeter("loss", "acc1_true", "acc1_bezier", "loss_orig", "reg")
    progress = ctx["progress"]

    for images, labels in progress.track(valid_dataloader, description="Valid", remove=True):
        batch = (images.to(device), labels.to(device))
        metrics = valid_step(ctx, batch=batch, bezier_model=bezier_model, model=model)
        meter.update(metrics, n=len(batch[0]))

    return meter


def main(cfg, output_dir):
    is_master = dist.is_master_process()
    is_distributed = dist.is_distributed()

    if is_master and is_distributed and not is_xla:
        dev_utils.setup_logger(None, output_dir, suppress=[torch, nxcl, nxml])
    elif not is_master and is_xla:
        root_logger = logging.getLogger(None)
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        root_logger.addHandler(logging.NullHandler(logging.DEBUG))

    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)

    logging.info(f"Distributed: {is_distributed}")
    logging.info(f"World size: {dist.get_world_size()}")
    logging.info(f"Device type: {'TPU' if is_xla else 'GPU'}")

    device = xm.xla_device() if is_xla else torch.device(type="cuda", index=dist.get_local_rank())

    # PyTorch config
    if not is_xla:
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN_DETERMINISTIC

    # build dataloaders
    dataloaders = build_dataloaders(
        cfg,
        root="./datasets",
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        is_distributed=dist.is_distributed(),
        persistent_workers=True,
    )
    train_dataloader = dataloaders["dataloader"]
    valid_dataloader = dataloaders["val_loader"]

    log_str = tabulate([
        (name, len(dataloader.dataset), len(dataloader), dataloader.batch_size)
        for name, dataloader in zip(["Train", "Valid"], [train_dataloader, valid_dataloader])
    ], headers=["Key", "# Examples", "# Batches", "Batch Size"]) + "\n"

    if is_xla:
        train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
        valid_dataloader = pl.MpDeviceLoader(valid_dataloader, device)

    logging.debug("Dataloaders:")
    for line in log_str.split("\n"):
        logging.debug(line)

    # Build bezier model
    bezier_model = build_model(ConfigDict(MODEL=cfg.BEZIER_MODEL))
    bezier_ckpt = torch.load(cfg.BEZIER_MODEL.CHECKPOINT, map_location=bezier_model.device)
    bezier_model.load_state_dict(bezier_ckpt)
    bezier_model.eval()

    logging.debug("Bezier model:")
    for line in str(bezier_model).split("\n"):
        logging.debug(line)

    # Build model
    model = build_bridge(cfg)

    logging.debug("Model:")
    for line in str(model).split("\n"):
        logging.debug(line)

    bezier_model.to(device)
    bezier_model = create_ddp_model(bezier_model)

    model.to(device)
    model = create_ddp_model(model)

    # Build optimizer and scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    if cfg.SOLVER.MIXUP_ALPHA > 0.:
        distribution = torch.distributions.Beta(cfg.SOLVER.MIXUP_ALPHA, cfg.SOLVER.MIXUP_ALPHA)
    else:
        distribution = None

    # Set helper variables
    valid_epochs = (
        [1]
        + list(range(0, cfg.SOLVER.NUM_EPOCHS - cfg.SOLVER.VALID_FINALE + 1, cfg.SOLVER.VALID_FREQUENCY))
        + list(range(cfg.SOLVER.NUM_EPOCHS - cfg.SOLVER.VALID_FINALE, cfg.SOLVER.NUM_EPOCHS + 1, 1))
    )
    num_batch = len(train_dataloader)
    logging_batches = list(range(0, num_batch + 1, num_batch // cfg.LOG_FREQUENCY)) + [num_batch]
    ne_fmt = f"{len(str(cfg.SOLVER.NUM_EPOCHS))}d"
    nb_fmt = f"{len(str(num_batch))}d"

    best_acc1, best_acc1_epoch = 0.0, -1
    best_loss, best_loss_epoch = float("inf"), -1

    # Train loop
    with Progress(disable=(not is_master), speed_estimate_period=600, refresh_per_second=1) as progress:
        for epoch_idx in progress.trange(1, cfg.SOLVER.NUM_EPOCHS + 1, description="Epoch"):

            ctx = dict(
                logging_batches=logging_batches, nb_fmt=nb_fmt,
                epoch_header=f"[Epoch {epoch_idx:{ne_fmt}}/{cfg.SOLVER.NUM_EPOCHS:{ne_fmt}}]",
                progress=progress, epoch_idx=epoch_idx, is_distributed=is_distributed,
                config=cfg, distribution=distribution,
            )

            train_meter = train_epoch(ctx, bezier_model, model, optimizer, scheduler, train_dataloader, device)

            logging.info(
                f"[Epoch {epoch_idx:{ne_fmt}}/{cfg.SOLVER.NUM_EPOCHS:{ne_fmt}}] "
                + f"Train Loss {train_meter.loss:9.6f}  "
                + (f"Loss/-reg {train_meter.loss_orig:.5f}  " if cfg.SOLVER.REGULARIZATION > 0. else "")
                + (f"Reg {train_meter.reg:.5f}  " if cfg.SOLVER.REGULARIZATION > 0. else "")
                + f"Acc/True {train_meter.acc1_true * 100:6.2f}  "
                + f"Acc/Bezier {train_meter.acc1_bezier * 100:6.2f}"
            )

            if epoch_idx not in valid_epochs:
                continue

            valid_meter = valid_epoch(dict(progress=progress, config=cfg), bezier_model, model, valid_dataloader, device)

            logging.info(
                f"[Epoch {epoch_idx:{ne_fmt}}/{cfg.SOLVER.NUM_EPOCHS:{ne_fmt}}] "
                + f"Valid Loss {valid_meter.loss:9.6f}  "
                + (f"Loss/-reg {valid_meter.loss_orig:.5f}  " if cfg.SOLVER.REGULARIZATION > 0. else "")
                + (f"Reg {valid_meter.reg:.5f}  " if cfg.SOLVER.REGULARIZATION > 0. else "")
                + f"Acc/True {valid_meter.acc1_true * 100:6.2f}  "
                + f"Acc/Bezier {valid_meter.acc1_bezier * 100:6.2f}"
            )

            if is_best_loss := (valid_loss := valid_meter.loss) < best_loss:
                best_loss = valid_loss
                best_loss_epoch = epoch_idx

            if is_best_acc1 := (valid_acc1 := valid_meter.acc1_bezier) > best_acc1:
                best_acc1 = valid_acc1
                best_acc1_epoch = epoch_idx

            if is_master and (is_best_loss or is_best_acc1):
                state_dict = get_state_dict(model)

                if is_best_loss:
                    filename = os.path.join(output_dir, "best_loss.pt")
                    torch.save(state_dict, filename)
                    logging.debug(f"[Checkpoint] Saved {filename}")

                # if is_best_acc1:
                #     filename = os.path.join(output_dir, "best_acc1.pt")
                #     torch.save(state_dict, filename)
                #     logging.debug(f"[Checkpoint] Saved {filename}")

            synchronize("epoch")

    logging.info(
        "Summary:\n"
        f"    Best Loss: {best_loss:.6f} @ Epoch {best_loss_epoch}\n"
        f"    Best Acc1: {best_acc1*100:6.2f} @ Epoch {best_acc1_epoch}"
    )

    # finished
    logging.info("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file", type=str, required=True)
    args, rest_args = parser.parse_known_args()

    cfg: ConfigDict = load_config(args.config_file)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_config_arguments(parser, cfg, aliases={
        "SOLVER.NUM_EPOCHS": ["-ne", "--num-epochs"],
        "SOLVER.BATCH_SIZE": ["-bs", "--batch-size"],
        "SOLVER.OPTIMIZER.SGD.BASE_LR": ["-lr", "--base-lr"],
        "SOLVER.OPTIMIZER.SGD.WEIGHT_DECAY": ["-wd", "--weight-decay"],
        "SOLVER.OPTIMIZER.SGD.MOMENTUM": ["-mt", "--momentum"],
        "SOLVER.OPTIMIZER.SGD.NESTEROV": ["-nt", "--nesterov"],
        "SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR.WARMUP_EPOCHS": ["-we", "--warmup-epochs"],
        "BEZIER_MODEL.CHECKPOINT": ["-c", "--bezier-checkpoint"],
        "MODEL.REQUIRES": ["-r", "--requires"],
        "MODEL.BACKBONES": ["-b", "--backbones"],
        "MODEL.TYPE": ["-t", "--model-type"],
        "MODEL.SLIM_FACTOR": ["-s", "--slim-factor"],
        "SOLVER.MATCH": ["-m", "--match"],
        "SOLVER.MIXUP_ALPHA": ["-a", "--mixup-alpha"],
        "SOLVER.REGULARIZATION": ["-g", "--regularization"],
        "NUM_DEVICES": ["-nd", "--num-devices"],
    })
    parser.add_argument("--dev",  default=False, action="store_true")
    parser.add_argument("-o", "--output-dir",    dest="OUTPUT_DIR", required=True)
    args = parser.parse_args(rest_args)

    dev_mode = args.dev
    del args.dev

    cfg.update(vars(args))

    if cfg.BEZIER_MODEL.CHECKPOINT == "TO_BE_SET":
        raise ValueError("Please specify the checkpoint path of the bezier model via `--bezier-checkpoint`.")

    if cfg.NUM_DEVICES == -1:
        cfg.NUM_DEVICES = NUM_AVAILABLE_DEVICES

    log_name = dev_utils.get_experiment_name()

    if dev_mode:
        log_name = f"dev-{log_name}"

    if cfg.OUTPUT_DIR[-1] == "/":
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR[:-1]

    base_output_dir = cfg.OUTPUT_DIR.split("/")[0]

    output_dir = os.path.join(base_output_dir, "_", log_name)
    os.makedirs(output_dir, exist_ok=True)
    save_config(cfg, os.path.join(output_dir, "config.yaml"))

    dev_utils.setup_logger(None, output_dir, suppress=[torch, nxcl, nxml])
    logging.debug("python " + " ".join(sys.argv))

    if os.path.exists(cfg.OUTPUT_DIR):
        alternate_output_dir = cfg.OUTPUT_DIR + " " + log_name.split("-")[-1]
        logging.warning(
            "[yellow]"
            f"Output directory \"{cfg.OUTPUT_DIR}\" already exists. Use \"{alternate_output_dir}\" instead."
            "[/]"
        )
        cfg.OUTPUT_DIR = alternate_output_dir

    os.makedirs(os.path.dirname(cfg.OUTPUT_DIR), exist_ok=True)
    os.symlink(os.path.relpath(output_dir, os.path.dirname(cfg.OUTPUT_DIR)), cfg.OUTPUT_DIR)

    logging.debug("Full configs:")
    for k, v in cfg.items(flatten=True):
        logging.debug(f"    {k}: {v}")
    logging.info("Command line configs:")
    for k, v in vars(args).items():
        logging.info(f"    {k}: [cyan]{v}[/]")
    logging.info(f"Output directory: \"{output_dir}\", \"{cfg.OUTPUT_DIR}\"")

    try:
        dist.launch(
            main,
            args=(cfg, output_dir),
            num_local_devices=cfg.NUM_DEVICES,
            start_method=("fork" if is_xla else "spawn"),
        )
        code = 0
    except KeyboardInterrupt:
        logging.info("Interrupted")
        code = 1
    except Exception as e:
        logging.exception(e)
        code = 2

    exit(code)
