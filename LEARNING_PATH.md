# Learning Path

A step-by-step path from genetic algorithm (GA) basics to "using GA for machine
learning optimization", ending with a classic GA project: solving the TSP.

> **Teaching mode:** explanations are bilingual (中文 + English). **All code is
> written by the learner** — each lesson gives concepts, a spec, and hints; the
> learner implements it, then we review together.

## Environment setup (one-time)

```powershell
# 1. Create and activate the conda environment
conda create -y -n ga-ml python=3.11 numpy scikit-learn matplotlib pandas
conda activate ga-ml

# 2. Install GA-specific libraries
pip install -r requirements.txt

# 3. Verify
python -c "import numpy, sklearn, pygad, deap; print('Environment OK')"
```

> Note: JupyterLab is intentionally not installed. On Windows it often fails to
> extract due to the 260-character path limit (InvalidArchiveError). This project
> runs plain scripts with `python xxx.py`, so a notebook is not needed. If you
> want one later, install it separately with `pip install notebook`.

Each time you start, just run: `conda activate ga-ml`

## Lessons

| # | File                                | What you learn                                  |
| - | ----------------------------------- | ----------------------------------------------- |
| 1 | `lessons/01_ga_from_scratch.py`     | Hand-write a GA; understand the 5 core steps    |
| 2 | `lessons/02_ga_with_pygad.py`       | Redo Lesson 1 with PyGAD; see how a library helps |
| 3 | `lessons/03_feature_selection.py`   | Use GA for feature selection (first ML task)    |
| 4 | `lessons/04_hyperparam_tuning.py`   | Use GA to tune RandomForest hyperparameters     |
| 5 | `lessons/05_ga_vs_random.py`        | GA search vs random search — which wins?        |
| 6 | `lessons/06_tsp.py`                 | Capstone: GA solves the Travelling Salesman Problem |

## How to run

```powershell
conda activate ga-ml
python lessons/01_ga_from_scratch.py
```

## How to learn effectively

1. **Understand the concept** — read the bilingual explanation for the lesson.
2. **Write the code yourself** — implement it from the spec and hints.
3. **Run and observe** — look at the output and the plots.
4. **Experiment** — change the hyperparameters (population size, mutation rate,
   number of generations) and watch how the results change.

## Notes

- Plots a lesson writes (e.g. `lessons/01_result.png`) are git-ignored practice
  output — re-run the script to regenerate them.
- Lessons 2–6 are added as the course progresses; only the files present in
  `lessons/` have been reached so far.
