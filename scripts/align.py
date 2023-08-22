import os
import sys

sys.path.insert(0, os.getcwd())

import logging
import argparse
from copy import deepcopy

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
from bridge_net.modeling import build_model
from bridge_net import alignment

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
    # align_dataloader = dataloaders["trn_loader"]
    align_dataloader = dataloaders["aln_loader"]
    valid_dataloader = dataloaders["val_loader"]

    log_str = tabulate([
        (name, len(dataloader.dataset), len(dataloader), dataloader.batch_size)
        for name, dataloader in zip(["Align", "Valid"], [align_dataloader, valid_dataloader])
    ], headers=["Key", "# Examples", "# Batches", "Batch Size"]) + "\n"

    if is_xla:
        align_dataloader = pl.MpDeviceLoader(align_dataloader, device)
        valid_dataloader = pl.MpDeviceLoader(valid_dataloader, device)

    logging.debug("Dataloaders:")
    for line in log_str.split("\n"):
        logging.debug(line)

    # Build model
    model0 = build_model(cfg)
    model1 = deepcopy(model0)

    ckpt0 = torch.load(cfg.MODEL.CHECKPOINT_A, map_location="cpu")
    ckpt1 = torch.load(cfg.MODEL.CHECKPOINT_B, map_location="cpu")

    model0.load_state_dict(ckpt0)
    model1.load_state_dict(ckpt1)

    logging.debug("Model:")
    for line in str(model0).split("\n"):
        logging.debug(line)

    model0.to(device)
    model1.to(device)

    model0.eval()
    model1.eval()

    samples, correct0, correct1 = 0, 0, 0

    with torch.inference_mode():
        for image, label in valid_dataloader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            output0 = model0(image)["confidences"]
            output1 = model1(image)["confidences"]

            samples += label.size(0)
            correct0 += torch.sum(output0.argmax(dim=1) == label)
            correct1 += torch.sum(output1.argmax(dim=1) == label)

    logging.info(f"Acc (model 0): {correct0 / samples * 100:6.2f}")
    logging.info(f"Acc (model 1): {correct1 / samples * 100:6.2f}")

    logging.info("# Phase 1: Compute alignment")
    align_idxs, corr_mean = alignment.compute_model_alignment(model0, model1, align_dataloader)
    torch.save(corr_mean, f"{output_dir}/corr_mean.pt")
    torch.save(align_idxs, f"{output_dir}/align_idxs.pt")

    logging.info("# Phase 2: Align model")
    model1_aligned = alignment.align_model(model1, align_idxs)
    torch.save(model1_aligned.state_dict(), f"{output_dir}/best_acc1.pt")

    for (name, param), (name2, param2) in zip(model1.named_parameters(), model1_aligned.named_parameters()):
        assert name == name2, f"Not the same parameter: {name} != {name2}"
        if torch.all(torch.eq(param, param2)):
            logging.info(f"{name:45s} : unchanged")
        else:
            logging.info(f"{name:45s} : modified")

    logging.info("# Phase 3: Compute aligned correlation")
    _, corr_aligned_mean = alignment.compute_model_alignment(model0, model1_aligned, align_dataloader)
    torch.save(torch.asarray(corr_aligned_mean), f"{output_dir}/corr_aligned.pt")

    logging.info("Correlation before align:")
    logging.info([round(v.item(), 4) for v in corr_mean])
    logging.info("Correlation after align:")
    logging.info([round(v.item(), 4) for v in corr_aligned_mean])

    samples, correct1_aligned = 0, 0

    with torch.inference_mode():
        for image, label in valid_dataloader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            output1_aligned = model1_aligned(image)["confidences"]

            samples += label.size(0)
            correct1_aligned += torch.sum(output1_aligned.argmax(dim=1) == label)

    logging.info(f"Acc (model 0): {correct0 / samples * 100:6.2f}")
    logging.info(f"Acc (model 1): {correct1 / samples * 100:6.2f}")
    logging.info(f"Acc (aligned): {correct1_aligned / samples * 100:6.2f}")

    logging.info("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file", type=str, required=True)
    args, rest_args = parser.parse_known_args()

    cfg: ConfigDict = load_config(args.config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("-ca", "--model-checkpoint-a", dest="MODEL.CHECKPOINT_A")
    parser.add_argument("-cb", "--model-checkpoint-b", dest="MODEL.CHECKPOINT_B")
    parser.add_argument("--dev",  default=False,       action="store_true")
    parser.add_argument("-o", "--output-dir",          dest="OUTPUT_DIR", required=True)
    args = parser.parse_args(rest_args)

    dev_mode = args.dev
    del args.dev

    cfg.update(vars(args))

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
            num_local_devices=1,
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
