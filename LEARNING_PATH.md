# 学习路线 (Learning Path)

一步步从遗传算法基础走到「用 GA 做机器学习优化」。

## 环境准备（一次性）

```powershell
# 1. 创建并激活 conda 环境
conda create -y -n ga-ml python=3.11 numpy scikit-learn matplotlib pandas
conda activate ga-ml

# 2. 安装 GA 专用库
pip install -r requirements.txt

# 3. 验证
python -c "import numpy, sklearn, pygad, deap; print('环境 OK')"
```

> 注：这里特意没装 jupyterlab。它在 Windows 上常因「文件路径超过 260 字符」
> 解压失败（InvalidArchiveError）。本项目用 `python xxx.py` 直接跑脚本即可，
> 不需要 notebook。如果以后确实想用，可单独 `pip install notebook`。

之后每次开始学习，只需在终端运行：`conda activate ga-ml`

## 课程列表

| 课 | 文件 | 学什么 |
|----|------|--------|
| 1 | `lessons/01_ga_from_scratch.py` | 手写 GA，理解“选择/交叉/变异”五步骤 |
| 2 | `lessons/02_ga_with_pygad.py` | 用 PyGAD 库重做第 1 课，看库怎么简化 |
| 3 | `lessons/03_feature_selection.py` | 用 GA 做特征选择（机器学习第一站） |
| 4 | `lessons/04_hyperparam_tuning.py` | 用 GA 给 RandomForest 调超参 |
| 5 | `lessons/05_ga_vs_random.py` | GA 搜索 vs 随机搜索，谁更强？ |

## 运行方式

```powershell
conda activate ga-ml
python lessons/01_ga_from_scratch.py
```

## 怎么学最有效
1. **先跑通**：直接运行脚本，看输出和图。
2. **再读码**：脚本里每行都有中文注释，对照着理解。
3. **后动手**：改超参数（种群大小、变异率、代数），观察结果变化。
