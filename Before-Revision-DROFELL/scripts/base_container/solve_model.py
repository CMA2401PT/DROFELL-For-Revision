import os
import json
from typing import Any, List
import numpy as np
import matplotlib.pyplot as plt

with open(os.path.join(os.path.dirname(__file__), "model_solve.json"), "r") as f:
    data = json.load(f)
    data = data[len(data)//2:]
data: list[tuple[float, float, float, float]] = [(
    (float(y)/(10**6)), (float(cpu)/(10**6)), (float(cuda)/(10**6)), float(n)
)for (y, cpu, cuda, n) in data]

y_prime = []
A = []
for y, cpu, cuda, n in data:
    y_prime.append(y-cuda)
    A.append([cpu, n])

# w=(A^T A)^-1 A^T y
w, residual, rank, s = np.linalg.lstsq(np.array(A), np.array(y_prime), rcond=None)
alpha, beta = w

for y, cpu, cuda, n in data:
    print(f"actual load: {y}ms")
    estimate_load = cpu*alpha+cuda-n*beta
    print(f"estimate load: {estimate_load:.2f}ms")

print(alpha, beta)


def predict(alpha: float, beta: float, x1: float, x2: float, x3: float):
    return alpha * x1 + x2 + beta * x3


def plot_fit(data: list[tuple[float, float, float, float]], alpha: float, beta: float):
    y_true = []
    y_pred = []
    y_pred_alpha_0 = []
    y_pred_alpha_1 = []
    for y, x1, x2, x3 in data:
        y_true.append(y)
        y_pred.append(predict(alpha, beta, x1, x2, x3))
        y_pred_alpha_0.append(predict(0, beta, x1, x2, x3))
        y_pred_alpha_1.append(predict(1, beta, x1, x2, x3))

    # 画图：真实值 vs 预测值
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 7.2), dpi=330, sharex=True,
        gridspec_kw={'height_ratios': [8, 2]}
    )
    # 上图：散点图 + 理想拟合线
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal fit (y = ŷ)')
    ax1.scatter(y_true, y_pred, color='blue', label='Predicted', s=10)
    ax1.scatter(y_true, y_pred_alpha_0, color='green', label='Alpha=0', s=10)
    ax1.scatter(y_true, y_pred_alpha_1, color='orange', label='Alpha=1', s=10)
    ax1.set_ylabel('Predicted ŷ')
    ax1.set_title('Fit Quality: Predicted vs Actual')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')  # 只对上图设置

    # 下图：偏差图（ŷ - y）随实际值的变化

    def plot_offset(ax: Any, y_true: List[float], y_pred: List[float], color: str):
        estimate_off = sorted((t, p - t) for p, t in zip(y_pred, y_true))
        x_vals = [y for y, _ in estimate_off]
        y_vals = [v for _, v in estimate_off]
        ax.plot(x_vals, y_vals, color=color)

    plot_offset(ax2, y_true, y_pred, 'blue')
    plot_offset(ax2, y_true, y_pred_alpha_0, 'green')
    plot_offset(ax2, y_true, y_pred_alpha_1, 'orange')
    ax2.set_xlabel('Actual y')
    ax2.set_ylabel('ŷ - y')
    ax2.grid(True)

    # 设置x轴范围
    # ax2.set_xlim(30, 400)

    plt.tight_layout()
    plt.show()


plot_fit(data, alpha, beta)
plt.savefig(os.path.join(os.path.dirname(__file__),  "model_fit.png"))
