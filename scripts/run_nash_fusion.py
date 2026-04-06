#!/usr/bin/env python3
"""V8: Nash-Gradient Flow Multi-Agent Fusion"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.nash_gradient_fusion import NashGradientFusion, Agent
import numpy as np


def run_nash_fusion(baseline_pred, gbdt_full_pred, gbdt_target_pred, y_true):
    """运行Nash梯度流优化"""

    agents = [
        Agent("baseline", lambda: baseline_pred, weight=0.33),
        Agent("gbdt_full", lambda: gbdt_full_pred, weight=0.33),
        Agent("gbdt_target", lambda: gbdt_target_pred, weight=0.34)
    ]

    predictions = {
        "baseline": baseline_pred,
        "gbdt_full": gbdt_full_pred,
        "gbdt_target": gbdt_target_pred
    }

    optimizer = NashGradientFusion(agents, learning_rate=0.05)
    optimal_weights = optimizer.optimize(predictions, y_true, max_iters=100)

    return optimal_weights


if __name__ == "__main__":
    print("Nash-Gradient Flow Fusion Optimizer")
    print("基于多智能体博弈的模型融合")
