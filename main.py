"""Reproduce the six tables and two figures from the numerical section."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import importlib
import os
from pathlib import Path
import sys

# Use one numerical thread in each worker process.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

from core.experiments import collect_trials, run_trial
from core.figures import plot_overviews
from core.tables import write_tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="Number of numerical worker processes (default: up to 4).")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs",
                        help="Directory for tables.txt and the two figure PDFs.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    summaries, trials, pending = {}, {}, {}
    completed_trials = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for case in (1, 2):
            for experiment in (1, 2, 3, 4):
                key = (case, experiment)
                module = importlib.import_module(f"case{case}.ex{experiment}")
                pending[pool.submit(module.run)] = ("prepare", key)
                trials[key] = []
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                kind, key = pending.pop(future)
                result = future.result()
                if kind == "prepare":
                    summaries[key] = result["summary"]
                    for trial in result["trials"]:
                        pending[pool.submit(run_trial, trial)] = ("trial", key)
                    print(f"Case{key[0]}/EX{key[1]}: clean data ready", flush=True)
                else:
                    trials[key].append(result)
                    completed_trials += 1
                    trial, fit = result
                    detail = trial.label or f"noise={trial.noise_level:g}, repeat={trial.repeat + 1}"
                    if trial.experiment == 3:
                        detail += f", selected={fit['selected_model']}"
                    else:
                        detail += f", inputs={trial.strategy}"
                    print(f"Trial {completed_trials}: Case{key[0]}/EX{key[1]}, {detail}", flush=True)
    for key in summaries:
        summaries[key] = collect_trials(summaries[key], trials[key], key[1])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_overviews(summaries, output)
    write_tables(summaries, output / "tables.txt")
    print("Completed: tables.txt", flush=True)
    print("Completed: case1_ex4_dual.pdf", flush=True)
    print("Completed: case2_ex4_dual.pdf", flush=True)


if __name__ == "__main__":
    main()
