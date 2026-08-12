# Genetic Algorithm ML Playground

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

A hands-on, lesson-by-lesson course on using **genetic algorithms (GA)** for
machine learning optimization — from writing a GA by hand in ~100 lines of
NumPy, to feature selection, hyperparameter tuning, and a Travelling Salesman
capstone.

用遗传算法解决机器学习优化问题的**渐进式实战课程**：从手写 GA 开始，逐步走到
特征选择、超参数调优，最后用 GA 求解旅行商问题（TSP）。

## Why this repo / 这个项目适合谁

- You want to *understand* GA mechanics (selection, crossover, mutation,
  elitism), not just call a library.
- You learn best by writing the code yourself: **every lesson ships as a spec
  plus TODO skeleton — the learner implements it**, then the solution is
  reviewed and refined together.
- Explanations are bilingual (中文 + English); code and comments are English.

## Quickstart

```powershell
# 1. Create and activate the environment
conda create -y -n ga-ml python=3.11 numpy scikit-learn matplotlib pandas
conda activate ga-ml

# 2. Install the GA libraries
pip install -r requirements.txt

# 3. Verify
python -c "import numpy, sklearn, pygad, deap; print('Environment OK')"
```

Then run any lesson:

```powershell
python lessons/01_ga_from_scratch.py
```

See [LEARNING_PATH.md](LEARNING_PATH.md) for setup notes, the full curriculum,
and how to get the most out of each lesson.

## Curriculum

| # | Lesson | What you learn |
| - | ------ | -------------- |
| 1 | [`lessons/01_ga_from_scratch.py`](lessons/01_ga_from_scratch.py) | Hand-write a GA; the 5 core steps (init → evaluate → select → crossover → mutate) |
| 2 | `lessons/02_ga_with_pygad.py` | Redo Lesson 1 with PyGAD; see what a library buys you |
| 3 | `lessons/03_feature_selection.py` | GA for feature selection — the first real ML task |
| 4 | `lessons/04_hyperparam_tuning.py` | GA to tune RandomForest hyperparameters |
| 5 | `lessons/05_ga_vs_random.py` | GA search vs. random search — which actually wins? |
| 6 | `lessons/06_tsp.py` | Capstone: GA solves the Travelling Salesman Problem |

Lessons are added as the course progresses; only the files present in
`lessons/` have been reached so far.

## Project structure

```
.
├── lessons/            # One script per lesson (spec + TODOs, filled in by the learner)
├── LEARNING_PATH.md    # Curriculum, environment setup, study method
├── requirements.txt    # Pip-installed GA libraries (conda covers the rest)
├── LICENSE             # MIT
└── README.md
```

Plots produced by a lesson (e.g. `lessons/01_result.png`) are personal practice
output and are git-ignored — run the script to regenerate them.

## Tools

Python 3.11 · NumPy · scikit-learn · matplotlib · [PyGAD](https://pygad.readthedocs.io/) · [DEAP](https://deap.readthedocs.io/)

## License

Released under the [MIT License](LICENSE).
