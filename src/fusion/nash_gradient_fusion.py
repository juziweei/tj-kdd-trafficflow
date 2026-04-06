"""Multi-Agent Nash-Gradient Flow for Model Fusion Optimization.

基于Nash均衡的多智能体博弈框架，将不同预测模型视为竞争智能体。
参考: Nash Policy Gradient & Utility-Taking Gradient Dynamics
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class Agent:
    """单个预测模型智能体"""
    name: str
    predict_fn: Callable
    weight: float = 0.33
    utility: float = 0.0


class NashGradientFusion:
    """Nash梯度流融合优化器"""

    def __init__(self, agents: list[Agent], learning_rate: float = 0.01):
        self.agents = agents
        self.lr = learning_rate
        self.n_agents = len(agents)

    def compute_utility(self, agent_idx: int, predictions: dict, y_true: np.ndarray) -> float:
        """计算智能体效用（负MAPE）"""
        pred = predictions[self.agents[agent_idx].name]
        mape = np.mean(np.abs((y_true - pred) / (y_true + 1e-8))) * 100
        return -mape

    def compute_nash_gradient(self, agent_idx: int, predictions: dict,
                             y_true: np.ndarray, weights: np.ndarray) -> float:
        """计算Nash梯度（考虑其他智能体反应）"""
        agent = self.agents[agent_idx]

        # 当前融合预测
        fused = sum(w * predictions[a.name] for w, a in zip(weights, self.agents))
        base_error = np.mean(np.abs((y_true - fused) / (y_true + 1e-8)))

        # 扰动当前智能体权重
        eps = 0.01
        weights_perturbed = weights.copy()
        weights_perturbed[agent_idx] += eps
        weights_perturbed /= weights_perturbed.sum()

        fused_perturbed = sum(w * predictions[a.name]
                             for w, a in zip(weights_perturbed, self.agents))
        perturbed_error = np.mean(np.abs((y_true - fused_perturbed) / (y_true + 1e-8)))

        # Nash梯度 = 边际效用改进
        gradient = -(perturbed_error - base_error) / eps
        return gradient

    def optimize(self, predictions: dict, y_true: np.ndarray,
                max_iters: int = 100, tol: float = 1e-4) -> np.ndarray:
        """Nash均衡优化"""
        weights = np.array([a.weight for a in self.agents])
        weights /= weights.sum()

        for iteration in range(max_iters):
            gradients = np.array([
                self.compute_nash_gradient(i, predictions, y_true, weights)
                for i in range(self.n_agents)
            ])

            # 投影梯度上升（保持权重和为1）
            weights_new = weights + self.lr * gradients
            weights_new = np.maximum(weights_new, 0.0)
            weights_new /= weights_new.sum()

            if np.linalg.norm(weights_new - weights) < tol:
                break

            weights = weights_new

        return weights
