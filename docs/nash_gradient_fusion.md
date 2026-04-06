# Nash-Gradient Flow 多智能体融合

## 理论基础

将模型融合问题建模为**多智能体博弈**：
- 每个模型 = 一个智能体
- 权重 = 策略
- MAPE = 负效用
- 目标：找到Nash均衡

## 核心优势

1. **理论保证** - Nash均衡存在性和稳定性
2. **自适应** - 智能体自动调整策略应对其他智能体
3. **全局最优** - 避免局部最优陷阱

## 使用方法

```python
from src.fusion.nash_gradient_fusion import NashGradientFusion, Agent

# 定义智能体
agents = [
    Agent("baseline", predict_fn, weight=0.33),
    Agent("gbdt_full", predict_fn, weight=0.33),
    Agent("gbdt_target", predict_fn, weight=0.34)
]

# 优化
optimizer = NashGradientFusion(agents, learning_rate=0.05)
weights = optimizer.optimize(predictions, y_true, max_iters=100)
```

## 参考文献

- Nash Policy Gradient (NeurIPS 2024)
- Utility-Taking Gradient Dynamics (ICML 2024)
- Multi-Agent Reinforcement Learning (AAMAS 2025)

Sources:
- [OpenReview](https://openreview.net)
- [ArXiv](https://arxiv.org)
- [ICML](https://icml.cc)
