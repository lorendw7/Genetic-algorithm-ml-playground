"""
第 1 课：从零手写遗传算法 (Genetic Algorithm from scratch)

目标：用最少的代码理解 GA 的 5 个核心步骤，不依赖任何第三方 GA 库。

问题：找到 x，使函数 f(x) = x * sin(10*pi*x) + 2 在区间 [-1, 2] 上最大。
      （这是一个有很多“山峰”的函数，普通求导很容易卡在局部最优，
       适合用 GA 这种“群体搜索”的方法。）

遗传算法的思路（模仿生物进化）：
  1. 初始化   : 随机生成一群“个体”(候选解)，组成种群 population
  2. 评估     : 用适应度函数 fitness 给每个个体打分（分数越高越好）
  3. 选择     : 分数高的个体更可能被选中当“父母”(优胜劣汰)
  4. 交叉     : 两个父母“基因”混合，产生后代 (crossover)
  5. 变异     : 后代基因有小概率随机改变，保持多样性 (mutation)
  重复 2~5 很多代(generation)，种群整体会越来越好。
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------- 0. 要优化的目标函数 ----------
def f(x):
    return x * np.sin(10 * np.pi * x) + 2.0


# ---------- 超参数（先用这些，之后你可以自己改着玩）----------
POP_SIZE = 50        # 种群里有多少个体
N_GEN = 100          # 进化多少代
X_LOW, X_HIGH = -1.0, 2.0   # 解的取值范围
MUTATION_RATE = 0.1  # 变异概率
MUTATION_SCALE = 0.1 # 变异时的扰动大小

rng = np.random.default_rng(42)  # 固定随机种子，保证结果可复现


# ---------- 1. 初始化种群 ----------
# 每个个体就是一个 x 值，随机撒在 [X_LOW, X_HIGH] 区间
population = rng.uniform(X_LOW, X_HIGH, size=POP_SIZE)


def fitness(pop):
    """适应度 = 目标函数值。值越大越好。"""
    return f(pop)


def select(pop, fit):
    """选择：锦标赛选择(tournament)。
    每次随机挑 3 个个体，留下其中最好的当父母。重复 POP_SIZE 次。"""
    selected = []
    for _ in range(len(pop)):
        idx = rng.integers(0, len(pop), size=3)   # 随机挑 3 个
        winner = idx[np.argmax(fit[idx])]         # 谁分高谁赢
        selected.append(pop[winner])
    return np.array(selected)


def crossover(parents):
    """交叉：把父母两两配对，后代取两者的加权平均(算术交叉)。"""
    children = parents.copy()
    for i in range(0, len(parents) - 1, 2):
        alpha = rng.random()                       # 混合比例
        children[i]     = alpha * parents[i] + (1 - alpha) * parents[i + 1]
        children[i + 1] = alpha * parents[i + 1] + (1 - alpha) * parents[i]
    return children


def mutate(children):
    """变异：每个后代有 MUTATION_RATE 的概率被加上一点随机噪声。"""
    for i in range(len(children)):
        if rng.random() < MUTATION_RATE:
            children[i] += rng.normal(0, MUTATION_SCALE)
            # 别让个体跑出取值范围
            children[i] = np.clip(children[i], X_LOW, X_HIGH)
    return children


# ---------- 主循环：开始进化 ----------
best_history = []   # 记录每一代的最优值，最后画图看收敛过程

for gen in range(N_GEN):
    fit = fitness(population)

    # 记录这一代最好的个体
    best_idx = np.argmax(fit)
    best_history.append(fit[best_idx])

    # 进化三步曲
    parents = select(population, fit)
    children = crossover(parents)
    population = mutate(children)

    if gen % 10 == 0:
        print(f"第 {gen:3d} 代  | 最优 x = {population[best_idx]:.4f}  | f(x) = {fit[best_idx]:.4f}")

# ---------- 输出最终结果 ----------
final_fit = fitness(population)
best_idx = np.argmax(final_fit)
print("\n==== 进化结束 ====")
print(f"找到的最优解 x = {population[best_idx]:.4f}")
print(f"对应函数值 f(x) = {final_fit[best_idx]:.4f}")
print("（真实最优值大约在 f(x) ≈ 3.85 附近）")


# ---------- 画图：看 GA 是怎么一步步变好的 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 左图：目标函数 + 最终种群分布
xs = np.linspace(X_LOW, X_HIGH, 500)
ax1.plot(xs, f(xs), label="f(x)")
ax1.scatter(population, final_fit, color="red", zorder=5, label="final population")
ax1.set_title("target function & final population")
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.legend()

# 右图：收敛曲线
ax2.plot(best_history)
ax2.set_xlabel("generation")
ax2.set_ylabel("best fitness")
ax2.set_title("convergence curve")

plt.tight_layout()
plt.savefig("lessons/01_result.png", dpi=100)
print("\n图已保存到 lessons/01_result.png，打开看看吧！")
