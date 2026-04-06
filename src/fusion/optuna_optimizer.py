"""Optuna-based hyperparameter optimization for fusion weights."""

from __future__ import annotations

import numpy as np
from typing import Callable

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def optimize_fusion_weights(
    objective_fn: Callable[[dict], float],
    n_trials: int = 100,
    n_branches: int = 3,
    timeout: int = 3600
) -> dict:
    """Optimize fusion weights using Bayesian optimization."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna required: pip install optuna")

    def objective(trial):
        weights = []
        for i in range(n_branches - 1):
            w = trial.suggest_float(f"weight_{i}", 0.0, 1.0)
            weights.append(w)

        total = sum(weights)
        if total > 1.0:
            weights = [w / total for w in weights]
        weights.append(1.0 - sum(weights))

        weight_dict = {
            "linear": weights[0],
            "gbdt_full": weights[1],
            "gbdt_target": weights[2]
        }
        return objective_fn(weight_dict)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best_params = study.best_params
    best_weights = [best_params[f"weight_{i}"] for i in range(n_branches - 1)]
    total = sum(best_weights)
    if total > 1.0:
        best_weights = [w / total for w in best_weights]
    best_weights.append(1.0 - sum(best_weights))

    return {
        "linear": best_weights[0],
        "gbdt_full": best_weights[1],
        "gbdt_target": best_weights[2],
        "best_value": study.best_value
    }
