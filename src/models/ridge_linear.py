"""A tiny ridge-regression model implemented with numpy only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RidgeLinearModel:
    feature_names: list[str]
    alpha: float = 5.0
    coef_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        x_mat = x[self.feature_names].to_numpy(dtype=float)
        y_vec = y.to_numpy(dtype=float)

        self.mean_ = x_mat.mean(axis=0)
        self.scale_ = x_mat.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0

        x_norm = (x_mat - self.mean_) / self.scale_
        ones = np.ones((x_norm.shape[0], 1), dtype=float)
        design = np.hstack([ones, x_norm])

        eye = np.eye(design.shape[1], dtype=float)
        eye[0, 0] = 0.0
        lhs = design.T @ design + self.alpha * eye
        rhs = design.T @ y_vec
        self.coef_ = np.linalg.solve(lhs, rhs)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Model is not fitted")

        x_mat = x[self.feature_names].to_numpy(dtype=float)
        x_norm = (x_mat - self.mean_) / self.scale_

        ones = np.ones((x_norm.shape[0], 1), dtype=float)
        design = np.hstack([ones, x_norm])
        preds = design @ self.coef_
        return np.clip(preds, a_min=0.0, a_max=None)
