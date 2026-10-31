"""Prepare the four experiments and their independent noise trials."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib

import numpy as np

from .backend import MODELS, SEEDS, Problem, parameters
from .inverse import recover, rms, sensitivity
from .optimal_design import make_optimal_design
from .protocol import BASE_LEVEL, DATA_LEVEL, SENSOR_STD, CorrectedProblem

NOISE_LEVELS = (1e-5, 1e-4, 1e-3)
REPEATS = 10
STRATEGIES = ("f_only", "g_only", "dual")
_CONTEXTS = OrderedDict()


@dataclass
class Trial:
    case: int
    experiment: int
    design: list
    observed: np.ndarray
    validation: list | None = None
    heldout: np.ndarray | None = None
    strategy: str = "all"
    noise_level: float | None = None
    repeat: int = 0
    label: str = ""


def _signature(rows):
    if rows is None:
        return None
    coefficients = np.array([[row["f"], row["g"]] for row in rows], dtype=float)
    return hashlib.sha256(coefficients.tobytes()).digest()


def _context(case, model, design, validation=None):
    key = (case, model, _signature(design), _signature(validation))
    if key not in _CONTEXTS:
        problem = CorrectedProblem(case, model)
        _CONTEXTS[key] = (
            problem, problem.prepare(design),
            problem.prepare(validation) if validation is not None else None,
        )
        if len(_CONTEXTS) > 6:
            _CONTEXTS.popitem(last=False)
    _CONTEXTS.move_to_end(key)
    return _CONTEXTS[key]


def prepare_experiment(case, experiment):
    """Compute clean table entries and construct paired noise trials."""
    print(f"Case{case}/EX{experiment}: select boundary inputs", flush=True)
    model, seed = MODELS[experiment], SEEDS[experiment]
    truth = parameters(model)[0]
    count = 48 if experiment == 1 else 96
    design, _ = make_optimal_design(case, experiment, count, seed)
    data_problem = Problem(case, model, DATA_LEVEL)
    summary, trials = {}, []
    if experiment == 3:
        validation, _ = make_optimal_design(case, experiment, 48, seed + 100000)
        observed = data_problem.measure(truth, data_problem.prepare(design))
        heldout = data_problem.measure(truth, data_problem.prepare(validation))
        control = Problem(case, "quadratic", DATA_LEVEL)
        control_truth = parameters("quadratic")[0]
        trials.append(Trial(
            case, experiment, design,
            control.measure(control_truth, control.prepare(design)), validation,
            control.measure(control_truth, control.prepare(validation)),
            label="quadratic_control_comparison",
        ))
        trials.append(Trial(
            case, experiment, design, observed, validation, heldout,
            label="clean_cubic_comparison",
        ))
        rng = np.random.default_rng(seed + 200000)
        for level in NOISE_LEVELS:
            for repeat in range(REPEATS):
                train_noise = level * SENSOR_STD * rng.standard_normal(count)
                val_noise = level * SENSOR_STD * rng.standard_normal(48)
                trials.append(Trial(
                    case, experiment, design, observed + train_noise,
                    validation, heldout + val_noise,
                    noise_level=level, repeat=repeat,
                ))
    else:
        designs = design if experiment == 4 else {"all": design}
        base = Problem(case, model, BASE_LEVEL)
        corrected = CorrectedProblem(case, model)
        evaluation = Problem(case, model, DATA_LEVEL)
        fine_data = {}
        for strategy, rows in designs.items():
            print(f"Case{case}/EX{experiment}/{strategy}: clean inversion", flush=True)
            observed = data_problem.measure(truth, data_problem.prepare(rows))
            fine_data[strategy] = observed
            fit = recover(corrected, corrected.prepare(rows), observed, truth)
            fit.update(evaluation.field_errors(truth, fit["recovered"]))
            group = {"cross_finer": fit}
            if experiment == 2:
                raw = recover(base, base.prepare(rows), observed, truth)
                raw.update(evaluation.field_errors(truth, raw["recovered"]))
                group["cross_raw"] = raw
            if experiment == 4:
                group["sensitivity"] = sensitivity(base, base.prepare(rows), truth)
                summary.setdefault("strategies", {})[strategy] = group
            else:
                summary.update(group)
        if experiment == 4:
            rng = np.random.default_rng(seed + 200000)
            for level in NOISE_LEVELS:
                for repeat in range(REPEATS):
                    noise = level * SENSOR_STD * rng.standard_normal(count)
                    for strategy in STRATEGIES:
                        trials.append(Trial(
                            case, experiment, designs[strategy],
                            fine_data[strategy] + noise, strategy=strategy,
                            noise_level=level, repeat=repeat,
                        ))
    return {"case": case, "experiment": experiment, "summary": summary, "trials": trials}


def run_trial(trial):
    """Fit a noise trial or a clean nonlinear-order comparison."""
    if trial.experiment == 3:
        candidates = {}
        linear_q = None
        for model in ("linear", "quadratic", "cubic"):
            problem, train, validation = _context(
                trial.case, model, trial.design, trial.validation,
            )
            initial = None if model == "linear" else np.r_[linear_q, np.zeros(3)]
            fit = recover(problem, train, trial.observed, initial=initial)
            if model == "linear":
                linear_q = fit["recovered"]
            fit["validation_rms"] = rms(
                problem.measure(fit["recovered"], validation) - trial.heldout
            )
            candidates[model] = fit
        selected = min(candidates, key=lambda name: candidates[name]["validation_rms"])
        result = {"candidates": candidates, "selected_model": selected}
    else:
        model = MODELS[trial.experiment]
        problem, inputs, _ = _context(trial.case, model, trial.design)
        result = recover(problem, inputs, trial.observed, parameters(model)[0])
    return trial, result


def collect_trials(summary, completed, experiment):
    """Aggregate trials in seed order, independent of worker completion order."""
    completed = sorted(completed, key=lambda pair: (
        pair[0].noise_level or 0, pair[0].repeat, pair[0].strategy, pair[0].label,
    ))
    if experiment == 3:
        for trial, result in completed:
            if trial.label:
                summary[trial.label] = result
        summary["order_selection"] = []
        for level in NOISE_LEVELS:
            group = [result["selected_model"] for trial, result in completed
                     if trial.noise_level == level]
            if len(group) != REPEATS:
                raise RuntimeError("Incomplete nonlinear-order trials.")
            summary["order_selection"].append({
                "noise_level": level,
                "counts": {name: group.count(name) for name in ("linear", "quadratic", "cubic")},
            })
    elif experiment == 4:
        summary["noise"] = []
        for level in NOISE_LEVELS:
            for strategy in STRATEGIES:
                errors = [result["parameter_relative_RMS"] for trial, result in completed
                          if trial.noise_level == level and trial.strategy == strategy]
                if len(errors) != REPEATS:
                    raise RuntimeError("Incomplete boundary-input trials.")
                summary["noise"].append({
                    "noise_level": level, "strategy": strategy,
                    "mean_parameter_relative_RMS": float(np.mean(errors)),
                })
    return summary
