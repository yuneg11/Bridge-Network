import os
import math
from typing import Dict, Sequence, Optional

import re  # TODO: remove

import torch

from nxcl.rich import Progress
from nxcl.config import load_config

from .modeling import build_model, build_bridge
from .data import build_dataloaders


class FeatureIterator:
    def __init__(
        self,
        batch_size: int,
        features_0: Dict[str, torch.Tensor],
        features_1: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.batch_size = batch_size
        num_samples = -1

        for k, v in features_0.items():
            if num_samples == -1:
                num_samples = v.shape[0]
            elif num_samples != v.shape[0]:
                raise ValueError(f"Dimension mismatch for bezier_features_0['{k}']")

        if features_1 is not None:
            for k, v in features_1.items():
                if num_samples != v.shape[0]:
                    raise ValueError(f"Dimension mismatch for bezier_features_1['{k}']")

        self.num_batches = math.ceil(num_samples / batch_size)
        self.features_0 = features_0
        self.features_1 = features_1

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self.features_1 is None:
            chunked_0 = {k: torch.split(v, self.batch_size, dim=0) for k, v in self.features_0.items()}
            for i in range(self.num_batches):
                yield {k: v[i] for k, v in chunked_0.items()}
        else:
            chunked_0 = {k: torch.split(v, self.batch_size, dim=0) for k, v in self.features_0.items()}
            chunked_1 = {k: torch.split(v, self.batch_size, dim=0) for k, v in self.features_1.items()}
            for i in range(self.num_batches):
                yield {k: v[i] for k, v in chunked_0.items()}, {k: v[i] for k, v in chunked_1.items()}


def get_labels(
    dataset: str,
    split: str = "tst",
    verbose: bool = False,
    overwrite: bool = False,
    cache: bool = True,
):
    if dataset not in ("cifar10", "cifar100", "tin200", "in1k"):
        raise ValueError(f"Unknown dataset: {dataset}")

    if verbose:
        print(f"Loading labels: {dataset}")

    cache_path = os.path.join("./.cache", dataset, "data", split)

    if not os.path.exists(os.path.join(cache_path, "labels.pt")) or overwrite:
        if verbose:
            print("Not cached:", ["labels"])

        cfg = load_config(f"configs/{dataset}/base.yaml")
        dataloader = build_dataloaders(cfg, root="./datasets")[f"{split}_loader"]

        label_list = []

        with Progress(disable=not verbose) as progress:
            for _, labels in progress.track(dataloader, description="Evaluating"):
                label_list.append(labels)

        labels = torch.cat(label_list, dim=0)

        if cache:
            os.makedirs(cache_path, exist_ok=True)
            torch.save(labels, os.path.join(cache_path, "labels.pt"))

    else:
        labels = torch.load(os.path.join(cache_path, "labels.pt"))

    return labels


def _get_model(
    model_id: str,
    filename: str = "best_acc1.pt",
    verbose: bool = False,
    device: str = "cuda",
    _type: str = "base",
    **model_kwargs,
):
    # TODO: support model_id="cifar10" etc.

    if model_id.startswith("configs"):
        output_base, model_id = os.path.split(model_id)
        config_file = f"{_type}.yaml"
        load_state_dict = False
    elif "/" not in model_id:
        output_base = "configs"
        config_file = f"{_type}.yaml"
        load_state_dict = False
    elif model_id.startswith("outs"):
        output_base, *model_id_list = model_id.split("/")
        model_id = "/".join(model_id_list)
        config_file = "config.yaml"
        load_state_dict = True
    else:
        output_base = "outs"
        config_file = "config.yaml"
        load_state_dict = True

    if verbose:
        print(f"Loading {_type} model: {model_id}")

    cfg = load_config(os.path.join(output_base, model_id, config_file))
    for k, v in model_kwargs.items():
        cfg.MODEL[k] = v

    if _type == "base" or _type == "bezier":
        model = build_model(cfg)
    elif _type == "bridge":
        model = build_bridge(cfg)
    else:
        raise ValueError(f"Unknown model type: {_type}")

    if load_state_dict:
        model.load_state_dict(torch.load(os.path.join(output_base, model_id, filename), map_location="cpu"))

    model.eval()
    model.to(device)

    return model


def get_base_model(
    model_id: str,
    filename: str = "best_acc1.pt",
    verbose: bool = False,
    device: str = "cuda",
    **model_kwargs,
):
    return _get_model(model_id, filename, verbose, device, "base", **model_kwargs)


def get_bezier_model(
    model_id: str,
    filename: str = "best_acc1.pt",
    verbose: bool = False,
    device: str = "cuda",
    **model_kwargs,
):
    return _get_model(model_id, filename, verbose, device, "bezier", **model_kwargs)


def get_bridge_model(
    model_id: str,
    filename: str = "best_loss.pt",
    verbose: bool = False,
    device: str = "cuda",
    **model_kwargs,
):

    return _get_model(model_id, filename, verbose, device, "bridge", **model_kwargs)


def _get_flops_and_params(
    model_id: str,
    verbose: bool = False,
    device: str = "cuda",
    _type: str = "base",
    **model_kwargs,
):
    pass

    # model = _get_model(model_id, None, verbose, device, _type, **model_kwargs)

    # if model_id.startswith("configs"):
    #     output_base, model_id = os.path.split(model_id)
    #     config_file = f"{_type}.yaml"
    #     load_state_dict = False
    # elif "/" not in model_id:
    #     output_base = "configs"
    #     config_file = f"{_type}.yaml"
    #     load_state_dict = False
    # elif model_id.startswith("outs"):
    #     output_base, *model_id_list = model_id.split("/")
    #     model_id = "/".join(model_id_list)
    #     config_file = "config.yaml"
    #     load_state_dict = True
    # else:
    #     output_base = "outs"
    #     config_file = "config.yaml"
    #     load_state_dict = True

    # if verbose:
    #     print(f"Loading {_type} model: {model_id}")

    # cfg = load_config(os.path.join(output_base, model_id, config_file))
    # for k, v in model_kwargs.items():
    #     cfg.MODEL[k] = v

    # if _type == "base" or _type == "bezier":
    #     model = build_model(cfg)
    # elif _type == "bridge":
    #     model = build_bridge(cfg)
    # else:
    #     raise ValueError(f"Unknown model type: {_type}")

    # if load_state_dict:
    #     model.load_state_dict(torch.load(os.path.join(output_base, model_id, filename), map_location="cpu"))

    # model.eval()
    # model.to(device)

    # return model



@torch.inference_mode()
def get_base_features(
    model_id: str,
    names: Sequence[str] = ("logits",),
    filename: str = "best_acc1.pt",
    split: str = "tst",
    verbose: bool = False,
    overwrite: bool = False,
    device: str = "cuda",
    cache: bool = True,
):

    if model_id.startswith("outs"):
        output_base, *model_id_list = model_id.split("/")
        model_id = "/".join(model_id_list)
    else:
        output_base = "outs"

    if verbose:
        print(f"Loading base features: {model_id}")

    cache_path = os.path.join("./.cache", model_id, filename.replace(".pt", ""), split)
    features = {}

    if overwrite:
        not_cached_names = names
        cached_names = []
    else:
        not_cached_names = [name for name in names if not os.path.exists(os.path.join(cache_path, name + ".pt"))]
        cached_names = [name for name in names if name not in not_cached_names]

    if not_cached_names:
        if verbose:
            print("Not cached:", not_cached_names)

        log_path = os.path.join(output_base, model_id, "info.log")

        if not os.path.exists(log_path):
            raise RuntimeError(f"Not available model: {model_id}")

        with open(log_path) as log_file:
            finished = False
            for line in log_file.readlines()[-3:]:
                if "Finished" in line:
                    finished = True
                    break
            if not finished:
                raise ValueError("Model is not finished")

        cfg = load_config(os.path.join(output_base, model_id, "config.yaml"))
        model = get_base_model(f"{output_base}/{model_id}", filename, verbose, device)
        dataloader = build_dataloaders(cfg, root="./datasets")[f"{split}_loader"]
        features_list = {k: [] for k in not_cached_names}

        with Progress(disable=not verbose) as progress:
            for images, _ in progress.track(dataloader, description="Evaluating"):
                outputs = model(images.to(device))

                for k in not_cached_names:
                    features_list[k].append(outputs[k].cpu())

        evaluated_features = {k: torch.cat(v, dim=0) for k, v in features_list.items()}
        features.update(evaluated_features)

        if cache:
            os.makedirs(cache_path, exist_ok=True)

            for name, v in evaluated_features.items():
                torch.save(v, os.path.join(cache_path, name + ".pt"))

                if verbose:
                    print(f"Saved to {os.path.join(cache_path, name + '.pt')}")

    if verbose:
        print("Loading cached:", cached_names)

    for name in cached_names:
        features[name] = torch.load(os.path.join(cache_path, name + ".pt"), map_location="cpu")

    return features


@torch.inference_mode()
def get_bezier_features(
    model_id: str,
    names: Sequence[str] = ("logits",),
    filename: str = "best_acc1.pt",
    split: str = "tst",
    bezier_lambda: float = None,
    verbose: bool = False,
    overwrite: bool = False,
    device: str = "cuda",
    cache: bool = True,
):

    if ":" in model_id:
        if bezier_lambda is None:
            model_id, bezier_lambda = model_id.split(":")
            bezier_lambda = float(bezier_lambda)
        else:
            raise ValueError("bezier_lambda is already specified in model_id")
    elif bezier_lambda is None:
        bezier_lambda = 0.5

    # TODO: this is temporary fix to increase disk efficiency
    if bezier_lambda == 0.0 or bezier_lambda == 1.0:
        base_ids = re.findall(r"/\d\-\d/", model_id)[0][1:-1].split("-")

        if bezier_lambda == 0.0:
            base_id = base_ids[0]
        elif bezier_lambda == 1.0:
            base_id = base_ids[1]
        else:
            raise ValueError

        model_id = model_id.replace("/bezier", "/base")
        model_id = re.sub(r"/\d\-\d/\d", f"/{base_id}", model_id)

        return get_base_features(model_id, names, filename, split, verbose, overwrite, device, cache)

    if model_id.startswith("outs"):
        output_base, *model_id_list = model_id.split("/")
        model_id = "/".join(model_id_list)
    else:
        output_base = "outs"

    if verbose:
        print(f"Loading bezier features: {model_id}")

    cache_path = os.path.join("./.cache", model_id, filename.replace(".pt", "") + f":{bezier_lambda:g}", split)
    features = {}

    if overwrite:
        not_cached_names = names
        cached_names = []
    else:
        not_cached_names = [name for name in names if not os.path.exists(os.path.join(cache_path, name + ".pt"))]
        cached_names = [name for name in names if name not in not_cached_names]

    if not_cached_names:
        if verbose:
            print("Not cached:", not_cached_names)

        log_path = os.path.join(output_base, model_id, "info.log")

        if not os.path.exists(log_path):
            raise RuntimeError(f"Not available model: {model_id}")

        with open(log_path) as log_file:
            finished = False
            for line in log_file.readlines()[-3:]:
                if "Finished" in line:
                    finished = True
                    break
            if not finished:
                raise ValueError("Model is not finished")

        cfg = load_config(os.path.join(output_base, model_id, "config.yaml"))
        model = get_bezier_model(f"{output_base}/{model_id}", filename, verbose, device)
        dataloader = build_dataloaders(cfg, root="./datasets")[f"{split}_loader"]
        features_list = {k: [] for k in not_cached_names}

        with Progress(disable=not verbose) as progress:
            for images, _ in progress.track(dataloader, description="Evaluating", remove=True):
                outputs = model(images.to(device), bezier_lambda=bezier_lambda)

                for k in not_cached_names:
                    features_list[k].append(outputs[k].cpu())

        evaluated_features = {k: torch.cat(v, dim=0) for k, v in features_list.items()}
        features.update(evaluated_features)

        if cache:
            os.makedirs(cache_path, exist_ok=True)
            for name, v in evaluated_features.items():
                torch.save(v, os.path.join(cache_path, name + ".pt"))

                if verbose:
                    print(f"Saved to {os.path.join(cache_path, name + '.pt')}")

    if verbose:
        print("Loading cached:", cached_names)

    for name in cached_names:
        features[name] = torch.load(os.path.join(cache_path, name + ".pt"), map_location="cpu")

    return features


@torch.inference_mode()
def get_batch_ensemble_features(
    model_id: str,
    names: Sequence[str] = ("logits",),
    filename: str = "best_acc1.pt",
    split: str = "tst",
    ensemble_idx: int = None,
    verbose: bool = False,
    overwrite: bool = False,
    device: str = "cuda",
    cache: bool = True,
):

    if ":" in model_id:
        if ensemble_idx is None:
            model_id, ensemble_idx = model_id.split(":")
            ensemble_idx = int(ensemble_idx)
        else:
            raise ValueError("bezier_lambda is already specified in model_id")
    elif ensemble_idx is None:
        ensemble_idx = 0

    if model_id.startswith("outs"):
        output_base, *model_id_list = model_id.split("/")
        model_id = "/".join(model_id_list)
    else:
        output_base = "outs"

    if verbose:
        print(f"Loading base features: {model_id}")

    cache_path_fn = lambda idx: os.path.join("./.cache", model_id, filename.replace(".pt", "") + f":{idx}", split)
    cache_path = cache_path_fn(ensemble_idx)
    features = {}

    if overwrite:
        not_cached_names = names
        cached_names = []
    else:
        not_cached_names = [name for name in names if not os.path.exists(os.path.join(cache_path, name + ".pt"))]
        cached_names = [name for name in names if name not in not_cached_names]

    if not_cached_names:
        if verbose:
            print("Not cached:", not_cached_names)

        log_path = os.path.join(output_base, model_id, "info.log")

        if not os.path.exists(log_path):
            raise RuntimeError(f"Not available model: {model_id}")

        with open(log_path) as log_file:
            finished = False
            for line in log_file.readlines()[-3:]:
                if "Finished" in line:
                    finished = True
                    break
            if not finished:
                raise ValueError("Model is not finished")

        cfg = load_config(os.path.join(output_base, model_id, "config.yaml"))
        ensemble_size = cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE

        if ensemble_idx > ensemble_size:
            raise ValueError(f"ensemble_idx should be less than ensemble_size: {ensemble_size}")

        model = get_base_model(f"{output_base}/{model_id}", filename, verbose, device)
        dataloader = build_dataloaders(cfg, root="./datasets")[f"{split}_loader"]
        features_lists = [{k: [] for k in not_cached_names} for _ in range(ensemble_size)]

        with Progress(disable=not verbose) as progress:
            for images, _ in progress.track(dataloader, description="Evaluating"):
                outputs = model(images.to(device).repeat(ensemble_size, 1, 1, 1))

                for k in not_cached_names:
                    feature_split = torch.split(outputs[k].cpu(), images.size(0))
                    for i in range(ensemble_size):
                        features_lists[i][k].append(feature_split[i])

        evaluated_features_list = [{k: torch.cat(v, dim=0) for k, v in features_list.items()} for features_list in features_lists]
        features.update(evaluated_features_list[ensemble_idx])

        if cache:
            for idx, evaluated_features in enumerate(evaluated_features_list):
                idx_cache_path = cache_path_fn(idx)
                os.makedirs(idx_cache_path, exist_ok=True)

                for name, v in evaluated_features.items():
                    torch.save(v, os.path.join(idx_cache_path, name + ".pt"))

                    if verbose:
                        print(f"Saved to {os.path.join(idx_cache_path, name + '.pt')}")

    if verbose:
        print("Loading cached:", cached_names)

    for name in cached_names:
        features[name] = torch.load(os.path.join(cache_path, name + ".pt"), map_location="cpu")

    return features


@torch.inference_mode()
def get_mimo_features(
    model_id: str,
    names: Sequence[str] = ("logits",),
    filename: str = "best_acc1.pt",
    split: str = "tst",
    ensemble_idx: int = None,
    verbose: bool = False,
    overwrite: bool = False,
    device: str = "cuda",
    cache: bool = True,
):
    if ":" in model_id:
        if ensemble_idx is None:
            model_id, ensemble_idx = model_id.split(":")
            ensemble_idx = int(ensemble_idx)
        else:
            raise ValueError("bezier_lambda is already specified in model_id")
    elif ensemble_idx is None:
        ensemble_idx = 0

    if model_id.startswith("outs"):
        output_base, *model_id_list = model_id.split("/")
        model_id = "/".join(model_id_list)
    else:
        output_base = "outs"

    if verbose:
        print(f"Loading base features: {model_id}")

    cache_path_fn = lambda idx: os.path.join("./.cache", model_id, filename.replace(".pt", "") + f":{idx}", split)
    cache_path = cache_path_fn(ensemble_idx)
    features = {}

    if overwrite:
        not_cached_names = names
        cached_names = []
    else:
        not_cached_names = [name for name in names if not os.path.exists(os.path.join(cache_path, name + ".pt"))]
        cached_names = [name for name in names if name not in not_cached_names]

    if not_cached_names:
        if verbose:
            print("Not cached:", not_cached_names)

        log_path = os.path.join(output_base, model_id, "info.log")

        if not os.path.exists(log_path):
            raise RuntimeError(f"Not available model: {model_id}")

        with open(log_path) as log_file:
            finished = False
            for line in log_file.readlines()[-3:]:
                if "Finished" in line:
                    finished = True
                    break
            if not finished:
                raise ValueError("Model is not finished")

        cfg = load_config(os.path.join(output_base, model_id, "config.yaml"))
        ensemble_size = cfg.MODEL.MIMO.ENSEMBLE_SIZE

        if ensemble_idx > ensemble_size:
            raise ValueError(f"ensemble_idx should be less than ensemble_size: {ensemble_size}")

        model = get_base_model(f"{output_base}/{model_id}", filename, verbose, device)
        dataloader = build_dataloaders(cfg, root="./datasets")[f"{split}_loader"]
        features_lists = [{k: [] for k in not_cached_names} for _ in range(ensemble_size)]

        with Progress(disable=not verbose) as progress:
            for images, _ in progress.track(dataloader, description="Evaluating"):
                outputs = model(images.to(device).repeat(1, ensemble_size, 1, 1))

                for k in not_cached_names:
                    if k in ["logits", "confidences", "log_confidences"]:
                        feature_split = list(map(lambda t: t.squeeze(dim=1), torch.split(outputs[k].cpu(), 1, dim=1)))
                    else:
                        feature_split = [outputs[k].cpu()] * ensemble_size

                    for i in range(ensemble_size):
                        features_lists[i][k].append(feature_split[i])

        evaluated_features_list = [{k: torch.cat(v, dim=0) for k, v in features_list.items()} for features_list in features_lists]
        features.update(evaluated_features_list[ensemble_idx])

        if cache:
            for idx, evaluated_features in enumerate(evaluated_features_list):
                idx_cache_path = cache_path_fn(idx)
                os.makedirs(idx_cache_path, exist_ok=True)

                for name, v in evaluated_features.items():
                    torch.save(v, os.path.join(idx_cache_path, name + ".pt"))

                    if verbose:
                        print(f"Saved to {os.path.join(idx_cache_path, name + '.pt')}")

    if verbose:
        print("Loading cached:", cached_names)

    for name in cached_names:
        features[name] = torch.load(os.path.join(cache_path, name + ".pt"), map_location="cpu")

    return features


@torch.inference_mode()
def get_bridge_features(
    model_id: str,
    names: Sequence[str] = ("logits",),
    filename: str = "best_loss.pt",
    split: str = "tst",
    verbose: bool = False,
    overwrite: bool = False,
    device: str = "cuda",
    cache: bool = True,
):

    if model_id.startswith("outs"):
        output_base, *model_id_list = model_id.split("/")
        model_id = "/".join(model_id_list)
    else:
        output_base = "outs"

    if verbose:
        print(f"Loading bridge features: {model_id}")

    cache_path = os.path.join("./.cache", model_id, filename.replace(".pt", ""), split)
    features = {}

    if overwrite:
        not_cached_names = names
        cached_names = []
    else:
        not_cached_names = [name for name in names if not os.path.exists(os.path.join(cache_path, name + ".pt"))]
        cached_names = [name for name in names if name not in not_cached_names]

    if not_cached_names:
        if verbose:
            print("Not cached:", not_cached_names)

        log_path = os.path.join(output_base, model_id, "info.log")

        if not os.path.exists(log_path):
            raise RuntimeError(f"Not available model: {model_id}")

        with open(log_path) as log_file:
            finished = False
            for line in log_file.readlines()[-3:]:
                if "Finished" in line:
                    finished = True
                    break
            if not finished:
                raise ValueError("Model is not finished")

        cfg = load_config(os.path.join(output_base, model_id, "config.yaml"))
        model = get_bridge_model(f"{output_base}/{model_id}", filename, verbose, device)

        get_base_features = lambda bezier_lambda: get_bezier_features(
            model_id="/".join(cfg.BEZIER_MODEL.CHECKPOINT.split("/")[:-1]),
            filename=cfg.BEZIER_MODEL.CHECKPOINT.split("/")[-1],
            names=cfg.MODEL.REQUIRES,
            bezier_lambda=bezier_lambda,
            split=split,
            verbose=verbose,
            device=device,
        )

        if cfg.MODEL.TYPE == 1:
            cfg.setdefault("MODEL.BASE_IDX", 0)
            bezier_lambda = 0. if cfg.MODEL.BASE_IDX == 0 else 1.

            base0_features = get_base_features(bezier_lambda=bezier_lambda)
            # dataloader = FeatureIterator(cfg.SOLVER.BATCH_SIZE * 10, base0_features)
            dataloader = FeatureIterator(cfg.SOLVER.BATCH_SIZE * 2, base0_features)
        elif cfg.MODEL.TYPE == 2:
            base0_features = get_base_features(bezier_lambda=0.)
            base1_features = get_base_features(bezier_lambda=1.)
            #dataloader = FeatureIterator(cfg.SOLVER.BATCH_SIZE * 10, base0_features, base1_features)
            dataloader = FeatureIterator(cfg.SOLVER.BATCH_SIZE, base0_features, base1_features)

        features_list = {k: [] for k in not_cached_names}

        with Progress(disable=not verbose) as progress:
            for base_features in progress.track(dataloader, description="Evaluating"):
                if cfg.MODEL.TYPE == 1:
                    base_outputs_0 = base_features
                    outputs = model(torch.cat([base_outputs_0[r] for r in cfg.MODEL.REQUIRES], dim=1).to(device))
                elif cfg.MODEL.TYPE == 2:
                    base_outputs_0, base_outputs_1 = base_features
                    outputs = model(
                        torch.cat([base_outputs_0[r] for r in cfg.MODEL.REQUIRES], dim=1).to(device),
                        torch.cat([base_outputs_1[r] for r in cfg.MODEL.REQUIRES], dim=1).to(device),
                    )

                for k in not_cached_names:
                    features_list[k].append(outputs[k].cpu())

        evaluated_features = {k: torch.cat(v, dim=0) for k, v in features_list.items()}
        features.update(evaluated_features)

        if cache:
            os.makedirs(cache_path, exist_ok=True)
            for name, v in evaluated_features.items():
                torch.save(v, os.path.join(cache_path, name + ".pt"))

                if verbose:
                    print(f"Saved to {os.path.join(cache_path, name + '.pt')}")

    if verbose:
        print("Loading cached:", cached_names)

    for name in cached_names:
        features[name] = torch.load(os.path.join(cache_path, name + ".pt"), map_location="cpu")

    return features
