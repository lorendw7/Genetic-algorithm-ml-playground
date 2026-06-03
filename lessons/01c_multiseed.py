"""
第 1 课（严谨篇）：随机算法要看「平均表现」，不能信单次结果

上一节我们发现 pop=300 单次跑反而更差——那其实是「运气」。
GA 每次初始种群都是随机的，单次结果带很大偶然性。
正确的评估方法：同一配置，换很多个随机种子(seed)各跑一次，看：
    - mean  平均最终值（整体有多好）
    - std   标准差    （有多不稳定 / 看运气）
    - 成功率 多少次真正逼近了全局最优 3.85

我们直接复用上一节写好的 run_ga（不重复造轮子）。

运行：  conda run -n ga-ml python lessons/01c_multiseed.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------- 目标函数 + GA（与前面一致，独立放这里便于单独运行）----------
def f(x):
    return x * np.sin(10 * np.pi * x) + 2.0


X_LOW, X_HIGH = -1.0, 2.0
TRUE_BEST = 3.85
SUCCESS_THRESHOLD = 3.80   # 最终值 >= 3.80 就算“成功逼近全局最优”


def run_ga(pop_size, n_gen, mutation_rate, mutation_scale, seed):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(X_LOW, X_HIGH, size=pop_size)
    history = []
    for _ in range(n_gen):
        fit = f(pop)
        history.append(fit.max())
        parents = np.empty_like(pop)
        for i in range(pop_size):
            idx = rng.integers(0, pop_size, size=3)
            parents[i] = pop[idx[np.argmax(fit[idx])]]
        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            a = rng.random()
            children[i]     = a * parents[i] + (1 - a) * parents[i + 1]
            children[i + 1] = a * parents[i + 1] + (1 - a) * parents[i]
        for i in range(pop_size):
            if rng.random() < mutation_rate:
                children[i] += rng.normal(0, mutation_scale)
                children[i] = np.clip(children[i], X_LOW, X_HIGH)
        pop = children
    return np.array(history), f(pop).max()


# ---------- 要对比的配置 ----------
configs = [
    ("baseline (mut=0.1)",          dict(mutation_rate=0.1, mutation_scale=0.1, pop_size=50)),
    ("high mut rate (mut=0.5)",     dict(mutation_rate=0.5, mutation_scale=0.1, pop_size=50)),
    ("big pop (pop=300)",           dict(mutation_rate=0.1, mutation_scale=0.1, pop_size=300)),
    ("explore a lot",               dict(mutation_rate=0.4, mutation_scale=0.4, pop_size=200)),
]

N_SEEDS = 10           # 每个配置跑 10 次
N_GEN = 100
seeds = range(N_SEEDS)

plt.figure(figsize=(10, 6))
print(f"{'配置':28s} | {'平均±标准差':>16s} | {'最好':>6s} | {'最差':>6s} | 成功率")
print("-" * 80)

for name, params in configs:
    histories = []   # 每个 seed 一条收敛曲线
    finals = []      # 每个 seed 的最终最优值
    for s in seeds:
        hist, best = run_ga(n_gen=N_GEN, seed=s, **params)
        histories.append(hist)
        finals.append(best)

    histories = np.array(histories)      # 形状 (N_SEEDS, N_GEN)
    finals = np.array(finals)

    mean_curve = histories.mean(axis=0)  # 每代的平均最优
    std_curve = histories.std(axis=0)    # 每代的波动
    success_rate = (finals >= SUCCESS_THRESHOLD).mean() * 100

    print(f"{name:28s} | {finals.mean():7.3f} ± {finals.std():5.3f} "
          f"| {finals.max():6.3f} | {finals.min():6.3f} | {success_rate:4.0f}%")

    # 画平均曲线 + 误差带(均值 ± 标准差)
    line, = plt.plot(mean_curve, label=f"{name} (mean={finals.mean():.2f})")
    plt.fill_between(range(N_GEN),
                     mean_curve - std_curve,
                     mean_curve + std_curve,
                     alpha=0.15, color=line.get_color())

plt.axhline(TRUE_BEST, color="gray", linestyle="--", label=f"true best ~{TRUE_BEST}")
plt.xlabel("generation")
plt.ylabel("best fitness (mean over seeds)")
plt.title(f"Mean convergence over {N_SEEDS} seeds (shaded = ±1 std)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("lessons/01c_multiseed.png", dpi=100)
print(f"\n图已保存到 lessons/01c_multiseed.png（阴影越窄 = 越稳定）")
