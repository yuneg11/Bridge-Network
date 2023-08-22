from enum import Enum, auto as enum_auto
from typing import Generator, Union

import os
import sys
import shutil
import argparse

from argparse import ArgumentParser
from pathlib import Path
from functools import cached_property
from collections import defaultdict

# Colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
GRAY = "\033[0;90m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"


def read_last_lines(filename: os.PathLike, n: int = 1):
    # https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python

    num_newlines = 0

    with open(filename, "rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b"\n":
                    num_newlines += 1
        except OSError:
            f.seek(0)

        last_lines = [line.decode().rstrip() for line in f.readlines()]

    return last_lines


class ExperimentStatus(Enum):
    FINISHED = enum_auto()
    RUNNING = enum_auto()
    BROKEN = enum_auto()
    INTERRUPTED = enum_auto()
    ERROR = enum_auto()
    INCOMPLETE = enum_auto()
    UNKNOWN = enum_auto()


class Experiment:
    def __init__(self, symlink: os.PathLike):
        self._symlink = Path(symlink)

        if not self._symlink.is_symlink():
            raise ValueError(f"Path {symlink} is not a symbolic link")

    @property
    def symlink(self):
        return self._symlink

    @cached_property
    def store(self):
        return self._symlink.resolve()

    @cached_property
    def id(self):
        return self.store.name

    @property
    def name(self):
        return str(self._symlink)

    @cached_property
    def root(self):
        return self.store.parent.parent

    @cached_property
    def status(self) -> ExperimentStatus:
        if (
            not self.store.exists()
            or not self.store.is_dir()
            or not self.store.joinpath("info.log").exists()
        ):
            return ExperimentStatus.BROKEN

        elif not self.store.joinpath("info.log").is_file():
            return ExperimentStatus.UNKNOWN

        else:
            last_lines = read_last_lines(self.store.joinpath("info.log"), n=2)
            if any(["Finished" in line for line in last_lines]):
                return ExperimentStatus.FINISHED
            elif any(["Interrupted" in line for line in last_lines]):
                return ExperimentStatus.INTERRUPTED
            elif any([("Error" in line or "Exception" in line) for line in last_lines]):
                return ExperimentStatus.ERROR
            else:
                # TODO: Check if the experiment is running or incomplete
                #       This can be done by checking the last log time in the info.log
                #       and comparing it to the current time.
                #       If the difference is greater than a threshold, then the experiment
                #       is considered incomplete.
                #       For now, we assume that the experiment is running.
                return ExperimentStatus.RUNNING

    def __str__(self):
        return f"{self.symlink} -> {self.store}"


def iter_experiments(
    path: os.PathLike,
) -> Generator[Experiment, None, None]:
    # Each experiment is a symbolic link to a experiment store
    # The experiment store is a directory containing the experiment data.
    path = Path(path)

    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")

    for child in path.iterdir():
        if child.is_symlink():
            yield Experiment(child)
        elif child.is_dir() and child.name != "_":
            yield from iter_experiments(child)


def iter_experiment_stores(
    path: os.PathLike,
) -> Generator[Union[Path, None], None, None]:
    path = Path(path)

    if not path.joinpath("_").exists() or not path.joinpath("_").is_dir():
        raise ValueError(f"Path {path} is not a valid experiment root")

    for child in path.joinpath("_").iterdir():
        if child.is_dir():
            yield child


def check_experiments(args):
    """
    Check the status of the experiments.
    """

    try:
        experiments = list(iter_experiments(args.path))  # May change to generator if this is too slow
        experiments.sort(key=lambda experiment: experiment.name)
    except ValueError:
        print(f"{RED}Path {args.path} is not a experiments{RESET}")
        return

    status_counts = defaultdict(int)

    # Print the experiments
    for experiment in experiments:
        status_counts[experiment.status] += 1

        if experiment.status == ExperimentStatus.FINISHED:
            if args.verbose:
                print(f"{experiment.name}: {GREEN}Finished{RESET}")
        elif experiment.status == ExperimentStatus.RUNNING:
            print(f"{experiment.name}: {CYAN}Running{RESET}")
        elif experiment.status == ExperimentStatus.INTERRUPTED:
            print(f"{experiment.name}: {YELLOW}Interrupted{RESET}")
        elif experiment.status == ExperimentStatus.ERROR:
            print(f"{experiment.name}: {RED}Error{RESET}")
        elif experiment.status == ExperimentStatus.BROKEN:
            print(f"{experiment.name}: {RED}Broken{RESET}")
        elif experiment.status == ExperimentStatus.INCOMPLETE:
            print(f"{experiment.name}: {YELLOW}Incomplete{RESET}")
        elif experiment.status == ExperimentStatus.UNKNOWN:
            print(f"{experiment.name}: {YELLOW}Unknown{RESET}")
        else:
            raise NotImplementedError(f"Unknown experiment status {experiment.status}")

    print("\nStatus summary:")

    # Print the status counts
    for status, count in sorted(status_counts.items(), key=lambda t: t[0].value):
        if status == ExperimentStatus.FINISHED:
            print(f"- Finished: {GREEN}{count}{RESET}")
        elif status == ExperimentStatus.RUNNING:
            print(f"- Running: {CYAN}{count}{RESET}")
        elif status == ExperimentStatus.INTERRUPTED:
            print(f"- Interrupted: {YELLOW}{count}{RESET}")
        elif status == ExperimentStatus.ERROR:
            print(f"- Error: {RED}{count}{RESET}")
        elif status == ExperimentStatus.BROKEN:
            print(f"- Broken: {RED}{count}{RESET}")
        elif status == ExperimentStatus.INCOMPLETE:
            print(f"- Incomplete: {YELLOW}{count}{RESET}")
        elif status == ExperimentStatus.UNKNOWN:
            print(f"- Unknown: {YELLOW}{count}{RESET}")
        else:
            raise NotImplementedError(f"Unknown experiment status {experiment.status}")

    # Delete the invalid experiments
    if args.delete or args.delete_all:
        if args.delete:
            invalid_experiments = [
                exp for exp in experiments if exp.status in (
                    ExperimentStatus.INTERRUPTED,
                    ExperimentStatus.ERROR,
                    ExperimentStatus.BROKEN,
                    ExperimentStatus.UNKNOWN,
                    ExperimentStatus.INCOMPLETE,
                )
            ]
        elif args.delete_all:
            invalid_experiments = [
                exp for exp in experiments if exp.status in (
                    ExperimentStatus.INTERRUPTED,
                    ExperimentStatus.ERROR,
                    ExperimentStatus.BROKEN,
                    ExperimentStatus.UNKNOWN,
                    ExperimentStatus.INCOMPLETE,
                    ExperimentStatus.RUNNING,
                )
            ]

    if (args.delete or args.delete_all) and len(invalid_experiments) > 0:
        print(f"\n{RED}Invalid experiments:{RESET}")

        prefix = Path(os.path.commonprefix([experiments[0].symlink.absolute(), experiments[0].store])).parent

        for i, experiment in enumerate(invalid_experiments, start=1):
            if experiment.status == ExperimentStatus.RUNNING:
                color, status = CYAN, "Running"
            elif experiment.status == ExperimentStatus.INTERRUPTED:
                color, status = YELLOW, "Interrupted"
            elif experiment.status == ExperimentStatus.ERROR:
                color, status = RED, "Error"
            elif experiment.status == ExperimentStatus.BROKEN:
                color, status = RED, "Broken"
            elif experiment.status == ExperimentStatus.INCOMPLETE:
                color, status = YELLOW, "Incomplete"
            elif experiment.status == ExperimentStatus.UNKNOWN:
                color, status = YELLOW, "Unknown"

            print(f"{i} - {color}{status}{RESET}: {experiment.symlink}")
            print(" " * len(f"{i} - {status}  ") + f"{GRAY}{experiment.store.relative_to(prefix)}{RESET}")

        print("")

        while True:
            print(f"Delete {len(invalid_experiments)} experiments? [y/N]:", end=" ", flush=True)
            response = input().lower()
            if response in ["y", "yes"]:
                break
            elif response in ["n", "no"] or response == "":
                return
            else:
                print("Please enter y or n")

        for experiment in invalid_experiments:
            try:
                experiment.symlink.unlink()
            except:
                print(f"{RED}Cannot remove {experiment.symlink}{RESET}")

            try:
                shutil.rmtree(experiment.store)
            except:
                print(f"{RED}Cannot remove {experiment.store}{RESET}")

        print(f"\n{GREEN}Deleted{RESET}")


def prune_experiment_stores(args):
    """
    Check the experiment stores and remove the ones that are not used.
    """

    try:
        experiment_stores = list(iter_experiment_stores(args.path))  # May change to generator if this is too slow
        experiment_stores.sort()
    except ValueError:
        print(f"{RED}Path {args.path} is not a valid experiment root{RESET}")
        return

    try:
        experiments = list(iter_experiments(args.path))  # May change to generator if this is too slow
        experiments.sort(key=lambda experiment: experiment.name)
    except ValueError:
        print(f"{RED}Path {args.path} is not a experiments{RESET}")
        return

    ref_dicts = {k.name: [] for k in experiment_stores}
    for experiment in experiments:
        ref_dicts[experiment.store.name].append(experiment.symlink)


    dangling_stores = []

    for exp_id, refs in ref_dicts.items():
        num_ref = len(refs)
        if num_ref == 0:
            dangling_stores.append(exp_id)
            print(f"{exp_id}: {RED}{num_ref} ref{RESET}")
        elif args.verbose:
            refs_str = ", ".join([str(ref.relative_to(args.path)) for ref in refs])
            color = GREEN if num_ref == 1 else BLUE
            print(f"{exp_id}: {color}{num_ref} ref(s){RESET} - {GRAY}{refs_str}{RESET}")


    if args.delete and len(dangling_stores) > 0:
        print(f"\n{RED}Dangling experiment stores:{RESET}")

        for i, store in enumerate(dangling_stores, start=1):
            print(f"{i} - {store}")

        print("")

        while True:
            print(f"Delete {len(dangling_stores)} experiment stores? [y/N]:", end=" ", flush=True)
            response = input().lower()
            if response in ["y", "yes"]:
                break
            elif response in ["n", "no"] or response == "":
                return
            else:
                print("Please enter y or n")

        prefix = Path(args.path).joinpath("_")

        for store in dangling_stores:
            store_path = prefix.joinpath(store)
            try:
                shutil.rmtree(store_path)
            except:
                print(f"{RED}Cannot remove {store_path}{RESET}")

        print(f"\n{GREEN}Deleted{RESET}")


def setup_check_parser(subparsers: argparse._SubParsersAction):
    parser: ArgumentParser = subparsers.add_parser("check", help="Check if the experiment is valid")
    parser.add_argument("path", help="Path to the experiments or sub experiments group")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose valid experiments")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--delete", action="store_true", help="Delete invalid experiments (broken / interrupted / error)")
    group.add_argument("-da", "--delete-all", action="store_true", help="Delete invalid and running experiments (broken / interrupted / error / running)")
    parser.set_defaults(func=check_experiments)


def setup_prune_parser(subparsers: argparse._SubParsersAction):
    parser: ArgumentParser = subparsers.add_parser("prune", help="Prune dangling experiment stores")
    parser.add_argument("path", help="Root path of the experiments")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose valid experiment stores")
    parser.add_argument("-d", "--delete", action="store_true", help="Delete dangling experiment stores (not linked to any experiment)")
    parser.set_defaults(func=prune_experiment_stores)


if __name__ == "__main__":
    parser = ArgumentParser(description="Experiment manager")
    subparsers = parser.add_subparsers(help="sub command")

    setup_check_parser(subparsers)
    setup_prune_parser(subparsers)

    args = parser.parse_args()
    func = args.func

    func(args)
