import os
import sys

sys.path.insert(0, os.getcwd())

import argparse
from collections import defaultdict

import yaml
from tabulate import tabulate

import torch
import numpy as np

from nxcl.rich import Progress
from nxml.torch.nn import functional as F

from bridge_net import store
from bridge_net.evaluations import (
    evaluate_acc,
    evaluate_nll,
    evaluate_ece,
    get_optimal_temperature,
    evaluate_bs,
)

torch.set_grad_enabled(False)


CALC_STD = {
    "acc": True,
    "cnll": True,
    "cece": True,
    "cbs": True,
    "nll": True,
    "ece": True,
    "bs": True,
    "flops": False,
    "params": False,
}


def parse_dataset(model_id):
    if "cifar100" in model_id:
        return "cifar100"
    elif "cifar10" in model_id:
        return "cifar10"
    elif "tin200" in model_id:
        return "tin200"
    elif "in1k" in model_id:
        return "in1k"
    else:
        raise ValueError(f"Unknown dataset for model_id: {model_id}")


def get_features(model_id, names=("logits",), split: str = "tst", cache: bool = True, overwrite: bool = False):
    kwargs = dict(names=names, split=split, verbose=False, cache=cache, overwrite=overwrite)

    if "base" in model_id:
        return store.get_base_features(model_id, **kwargs)
    elif "bezier" in model_id:
        return store.get_bezier_features(model_id, **kwargs)
    elif "batch" in model_id:
        return store.get_batch_ensemble_features(model_id, **kwargs)
    elif "mimo" in model_id:
        return store.get_mimo_features(model_id, **kwargs)
    elif "bridge" in model_id:
        return store.get_bridge_features(model_id, **kwargs)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def get_model(model_id):
    if "base" in model_id or "batch" in model_id or "mimo" in model_id:
        return store.get_base_model(model_id)
    elif "bezier" in model_id:
        return store.get_bezier_model(model_id)
    elif "bridge" in model_id:
        return store.get_bridge_model(model_id)
    elif model_id in ("cifar10", "cifar100", "tin200"):
        return store.get_base_model(model_id)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def get_flops_params(model_id):
    dataset = parse_dataset(model_id)

    if "base" in model_id:
        return store.get_base_model(model_id, base_id=dataset)
    elif "bezier" in model_id:
        return store.get_bezier_model(model_id, base_id=dataset)
    elif "bridge" in model_id:
        return store.get_bridge_model(model_id, base_id=dataset)
    elif model_id in ("cifar10", "cifar100", "tin200"):
        return store.get_base_model(model_id, base_id=dataset)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def eval_dee(group_metrics, dee_ref):
    dees = list(zip(*dee_ref))
    dee_vs = []

    for v, dee_list in zip(group_metrics, dees):
        inc = (dee_list[-1] - dee_list[0] > 0)  # True if increasing
        # dee_list = sorted(dee_list, reverse=inc)

        under_slope = dee_list[1] - dee_list[0]
        over_slope = dee_list[-1] - dee_list[-2]

        if inc:
            if v <= dee_list[0]:
                dee_v = 1 - ((dee_list[0] - v) / under_slope)
            elif v > dee_list[-1]:
                dee_v = (v - dee_list[-1]) / over_slope + len(dee_list)
            else:
                for i, dee_r in enumerate(dee_list[1:], 1):
                    if v <= dee_r:
                        dee_v = (i + 1) - ((dee_r - v) / (dee_r - dee_list[i-1]))
                        break
        else:
            if v >= dee_list[0]:
                dee_v = 1 - ((dee_list[0] - v) / under_slope)
            elif v < dee_list[-1]:
                dee_v = (v - dee_list[-1]) / over_slope + len(dee_list)
            else:
                for i, dee_r in enumerate(dee_list[1:], 1):
                    if v >= dee_r:
                        dee_v = (i + 1) - ((dee_r - v) / (dee_r - dee_list[i-1]))
                        break

        dee_vs.append(dee_v)

    return dee_vs


def eval_metrics(model_ids, cache: bool = True, overwrite: bool = False):

    if "cifar10/" in model_ids[0]:
        dataset = "cifar10"
    elif "cifar100/" in model_ids[0]:
        dataset = "cifar100"
    elif "tin200/" in model_ids[0]:
        dataset = "tin200"
    elif "in1k/" in model_ids[0]:
        dataset = "in1k"
    else:
        raise RuntimeError(f"Unknown dataset for model_ids: {model_ids[0]}")

    all_valid_confs = torch.stack([
        get_features(model_id, names=("confidences",), split="val", cache=cache, overwrite=overwrite)["confidences"]
        for model_id in model_ids
    ], dim=0)

    all_test_confs = torch.stack([
        get_features(model_id, names=("confidences",), split="tst", cache=cache, overwrite=overwrite)["confidences"]
        for model_id in model_ids
    ], dim=0)

    valid_confs = torch.mean(all_valid_confs, dim=0)
    test_confs  = torch.mean(all_test_confs,  dim=0)

    valid_labels = store.get_labels(dataset, split="val")
    test_labels  = store.get_labels(dataset, split="tst")

    opt_temp = get_optimal_temperature(valid_confs, valid_labels)
    test_cal_confs = F.softmax(torch.log(test_confs) / opt_temp, dim=1)

    test_acc  = evaluate_acc(test_confs,     test_labels) * 100.
    test_cnll = evaluate_nll(test_cal_confs, test_labels)
    test_cece = evaluate_ece(test_cal_confs, test_labels)
    test_cbs  = evaluate_bs(test_cal_confs,  test_labels)
    test_nll  = evaluate_nll(test_confs,     test_labels)
    test_ece  = evaluate_ece(test_confs,     test_labels)
    test_bs   = evaluate_bs(test_confs,      test_labels)

    return {
        "acc": test_acc,
        "cnll": test_cnll, "cece": test_cece, "cbs": test_cbs,
         "nll":  test_nll,  "ece":  test_ece,  "bs":  test_bs,
    }


def eval_group_metrics(group, dee=None, progress=None, cache: bool = True, overwrite: bool = False):
    group_metrics = defaultdict(list)

    it = group if progress is None else p.track(group, description="Samples", remove=True)

    for model_ids in it:
        metrics = eval_metrics(model_ids, cache=cache, overwrite=overwrite)

        for k, v in metrics.items():
            group_metrics[k].append(v)

    # # Use this to calculate the dee reference
    # print([round(v, 3) for v in group_metrics["acc"]])
    # print([round(v, 6) for v in group_metrics["cnll"]])

    if dee is not None:
        for k, dee_ref in dee.items():
            dee_v = eval_dee(group_metrics[k], dee_ref)
            group_metrics[f"dee_{k}"] = dee_v

    return {
        k: (np.mean(v), (np.std(v) if len(v) > 1 and CALC_STD[k] else -1))
        for k, v in group_metrics.items()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file", type=str, required=True)
    parser.add_argument("-e", "--dee-file",    type=str)
    parser.add_argument("-w", "--name-width",  type=int)
    parser.add_argument("-nc", "--no-cache",  action="store_true")
    parser.add_argument("-ow", "--overwrite", action="store_true")
    parser.add_argument("-o", "--output-path")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    table_headers = ["Group", "Acc (↑)", "cNLL (↓)", "cECE (↓)", "cBS (↓)"]
    # table_headers = ["Group", "Acc (↑)", "cNLL (↓)", "cECE (↓)", "cBS (↓)", "NLL (↓)", "ECE (↓)", "BS (↓)"]
    table_col_names = ["acc", "cnll", "cece", "cbs", ]
    table_col_fmts  = [".2f", ".3f", ".3f", ".3f", ]

    if args.dee_file is not None:
        with open(args.dee_file, "r") as f:
            dee = yaml.load(f, Loader=yaml.FullLoader)
        CALC_STD.update({f"dee_{k}": True for k in dee.keys()})
        table_col_names += [f"dee_{k}" for k in dee.keys()]
        table_col_fmts += [".3f" for _ in dee.keys()]
        table_headers += [f"DEE {k.upper()} (↑)" for k in dee.keys()]
    else:
        dee = None

    table_contents = []

    with Progress() as p:
        for group_name, group in p.track(cfg.items(), description="Groups", remove=True):
            metrics = eval_group_metrics(group, dee, progress=p, cache=not args.no_cache, overwrite=args.overwrite)

            if args.name_width is not None:
                group_name = f"{group_name:<{args.name_width}}"

            table_contents.append(
                [group_name] + [
                    f"{metrics[k][0]:{table_col_fmts[i]}}"
                    + (f" ± {metrics[k][1]:{table_col_fmts[i]}}" if metrics[k][1] > 0. else "")
                    for i, k in enumerate(table_col_names)
                ]
            )

    table = tabulate(table_contents, headers=table_headers, tablefmt="github")

    print(table)

    if args.output_path is not None:
        with open(args.output_path, "w") as f:
            f.write(table)
