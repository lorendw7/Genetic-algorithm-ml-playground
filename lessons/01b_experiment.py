"""
第 1 课（实验篇）：探索 vs 利用 —— 为什么 GA 会卡在局部最优？

第 1 课里 GA 卡在了 f(x)≈3.45，没找到真正的最优 ~3.85。
这节课我们把 GA 封装成一个函数 run_ga(...)，用不同参数跑很多次，
把收敛曲线画在一张图里对比，亲眼看清两个核心力量的拉扯：

  - 利用 (exploitation): 围绕当前最好的解精细搜索 → 收敛快，但容易困在局部山峰
  - 探索 (exploration) : 大胆乱跳、保持多样性     → 不容易困住，但收敛慢

调大「变异率 / 变异幅度 / 种群规模」= 增强探索。
我们就来验证：哪些参数能让 GA 跳出 3.45、爬上 3.85。

运行：  conda run -n ga-ml python lessons/01b_experiment.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------- 目标函数（和第 1 课一样）----------
def f(x):
    return x * np.sin(10 * np.pi * x) + 2.0


X_LOW, X_HIGH = -1.0, 2.0
TRUE_BEST = 3.85   # 这个函数在 [-1,2] 上的近似真实最优值，用作参考线


def run_ga(pop_size=50, n_gen=100, mutation_rate=0.1,
           mutation_scale=0.1, seed=42):
    """跑一次完整的遗传算法，返回 (每代最优值的历史, 最终找到的最优值)。

    所有参数都给了默认值，调用时只改你想实验的那个即可。
    每个个体就是一个 x 值。"""
    rng = np.random.default_rng(seed)
    pop = rng.uniform(X_LOW, X_HIGH, size=pop_size)
    history = []

    for _ in range(n_gen):
        fit = f(pop)
        history.append(fit.max())

        # --- 选择：锦标赛(每次随机挑3个，留最好的) ---
        parents = np.empty_like(pop)
        for i in range(pop_size):
            idx = rng.integers(0, pop_size, size=3)
            parents[i] = pop[idx[np.argmax(fit[idx])]]

        # --- 交叉：两两算术平均 ---
        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            a = rng.random()
            children[i]     = a * parents[i] + (1 - a) * parents[i + 1]
            children[i + 1] = a * parents[i + 1] + (1 - a) * parents[i]

        # --- 变异：按概率加噪声 ---
        for i in range(pop_size):
            if rng.random() < mutation_rate:
                children[i] += rng.normal(0, mutation_scale)
                children[i] = np.clip(children[i], X_LOW, X_HIGH)

        pop = children

    return history, f(pop).max()


# ---------- 实验：固定其他参数，只改一个，观察影响 ----------
# 每个配置写成 (图例名称, 参数字典)
experiments = [
    ("baseline (mut=0.1, scale=0.1, pop=50)", dict(mutation_rate=0.1, mutation_scale=0.1, pop_size=50)),
    ("high mutation rate (mut=0.5)",          dict(mutation_rate=0.5, mutation_scale=0.1, pop_size=50)),
    ("large mutation scale (scale=0.5)",      dict(mutation_rate=0.1, mutation_scale=0.5, pop_size=50)),
    ("big population (pop=300)",              dict(mutation_rate=0.1, mutation_scale=0.1, pop_size=300)),
    ("explore a lot (mut=0.4, scale=0.4, pop=200)", dict(mutation_rate=0.4, mutation_scale=0.4, pop_size=200)),
]

plt.figure(figsize=(10, 6))
print(f"{'配置':45s} | 最终最优 f(x)")
print("-" * 65)

for name, params in experiments:
    history, best = run_ga(n_gen=100, seed=42, **params)
    plt.plot(history, label=f"{name}  ->  {best:.3f}")
    print(f"{name:45s} | {best:.4f}")

# 画一条参考线：真实最优值
plt.axhline(TRUE_BEST, color="gray", linestyle="--", label=f"true best ~{TRUE_BEST}")

plt.xlabel("generation")
plt.ylabel("best fitness so far")
plt.title("Exploration vs Exploitation: how parameters affect GA")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("lessons/01b_experiment.png", dpi=100)
print("\n对比图已保存到 lessons/01b_experiment.png")
print("观察：探索力度大的配置，是不是更容易突破 3.45、接近虚线 3.85？")
