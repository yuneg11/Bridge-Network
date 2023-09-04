# Bridge Network

**Currently, we are refactoring the code. We will update README and the refactored code soon. This repository is not ready for use yet.**

This repository contains the implementation for _Traversing Between Modes in Function Space for Fast Ensembling_ (ICML 2023).

[EungGu Yun](https://github.com/yuneg11)\*, [Hyungi Lee](https://hyungi-lee.github.io/)\*, [Giung Nam](https://cs-giung.github.io)\*, [Juho Lee](https://juho-lee.github.io)

[[`Paper`](https://arxiv.org/abs/2306.11304)][[`ICML`](https://icml.cc/virtual/2023/poster/24321)][[`BibTeX`](#citation)]

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

Please see [giung2](https://github.com/cs-giung/giung2/tree/main/datasets) for the details and the preparation of the datasets for now.
We will add the details of the datasets soon.

## Usage

We need following steps to train a bridge network.
1. Train **base models**.
2. Train **bezier models** using the trained base models.
3. Train **bridge network** using the trained bezier models.

### Train base model

```bash
python scripts/train_base.py \
    -f, --config-file CONFIG_FILE \
    [-o, --output-dir OUTPUT_DIR] \
    [--dev] \
    [additional options]
```

You can change the values of the parameters in the config file using `--<config-key> <value>` options.
For example, if you want to change the `learning_rate` to 0.1, you can use `--SOLVER.OPTIMIZER.SGD.BASE_LR 0.1` or `-lr 0.1`, because `-lr` is registered as an alias of `--SOLVER.OPTIMIZER.SGD.BASE_LR` in `scripts/train_base.py`.

**NOTE**: The outputs are actually saved in `outs/_/<date>-<time>-<id>/` directory.
          The `--output-dir` option just makes the link to the directory.

#### See help

```bash
python scripts/train_base.py -f CONFIG_FILE --help
```

#### Example

- Basic case:

    ```bash
    python scripts/train_base.py \
        -f configs/cifar100/base.yaml \
        -o outs/cifar100/base/0
    ```

  - Use config file `configs/cifar100/base.yaml` to train a base model.
  - Save outputs under `outs/cifar100/base/0/`

### Train bezier model

```bash
python scripts/train_bezier.py \
    -f, --config-file CONFIG_FILE \
    [-o, --output-dir OUTPUT_DIR] \
    [--dev] \
    [additional options]
```

#### Example

- Basic case:

    ```bash
    python scripts/train_bezier.py \
        -f configs/cifar100/bezier.yaml \
        -ca outs/cifar100/base/0/best_acc1.pt \
        -cb outs/cifar100/base/1/best_acc1.pt \
        -o outs/cifar100/bezier/0-1/0
    ```

  - Use config file `configs/cifar100/bezier.yaml` to train a bezier model.
  - Use `outs/cifar100/base/0/best_acc1.pt` as the checkpoint of the first base model.
  - Use `outs/cifar100/base/1/best_acc1.pt` as the checkpoint of the second base model.
  - Save outputs under `outs/cifar100/bezier/0-1/0/`

### Train bridge model

```bash
python scripts/train_bridge.py \
    -f, --config-file CONFIG_FILE \
    [-o, --output-dir OUTPUT_DIR] \
    [--dev] \
    [additional options]
```

#### Example

- Basic case:

    ```bash
    python scripts/train_bridge.py \
        -f configs/cifar100/bridge.yaml \
        -c outs/cifar100/bezier/0-1/0/best_acc1.pt \
        -o outs/cifar100/bridge/0-1/0
    ```

  - Use config file `configs/cifar100/bridge.yaml` to train a bridge network.
  - Use `outs/cifar100/bezier/0-1/0/best_acc1.pt` as the checkpoint of the bezier model.
  - Save outputs under `outs/cifar100/bridge/0-1/0/`

## Acknowledgement

The model implementations are based on [giung2](https://github.com/cs-giung/giung2).\
Research supported with Cloud TPUs from [Google's TPU Research Cloud (TRC)](https://sites.research.google/trc/)

## License

See [LICENSE](LICENSE).

## Citation

```
@inproceedings{yun2023traversing,
    title     = {Traversing Between Modes in Function Space for Fast Ensembling},
    author    = {Yun, EungGu and Lee, Hyungi and Nam, Giung and Lee, Juho},
    booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML 2023)},
    year      = {2023},
}
```
