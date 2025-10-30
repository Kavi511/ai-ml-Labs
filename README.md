## AI & Machine Learning Practicals 

Central repository for AI/ML workshops and practicals, including datasets, notebooks, scripts, and reports. Each workshop is self‑contained and reproducible.

### Contents
- **Workshops overview**: Summary of all workshops and learning goals
- **Repository structure**: Folder layout and conventions
- **Environment setup**: How to install dependencies
- **How to run**: Notebooks and scripts
- **Datasets**: Where data lives and how to download
- **Results & reports**: Where outputs are saved
- **Reproducibility**: Seeds and notes
- **Academic integrity & license**: Usage/limits

---

## Workshops Overview

- **Workshop 01 — Intro to Python & NumPy**
  - Python refresher, control flow, functions
  - NumPy arrays, broadcasting, vectorization
  - Quick plotting with Matplotlib

- **Workshop 02 — Data Preprocessing & EDA**
  - Loading with Pandas, data types, summaries
  - Handling missing values, outliers
  - Scaling/normalization; visual EDA with Seaborn

- **Workshop 03 — Supervised Classification**
  - Train/validation/test split, baselines
  - k‑NN, Logistic Regression, Decision Trees
  - Evaluation with confusion matrix and ROC‑AUC

- **Workshop 04 — Regression Models**
  - Linear and Polynomial Regression
  - Regularization: Ridge/Lasso, bias–variance
  - Error analysis: MAE/MSE/RMSE

- **Workshop 05 — Model Evaluation**
  - Classification metrics: accuracy, precision/recall, F1
  - ROC/PR curves; threshold tuning
  - Regression metrics and diagnostic plots

- **Workshop 06 — Feature Engineering**
  - Encoding (one‑hot/ordinal), binning, scaling
  - Text/image feature basics; feature importance
  - Leakage pitfalls and good practices

- **Workshop 07 — Model Tuning & Cross‑Validation**
  - Pipelines, Grid/Random Search, k‑fold CV
  - Hyperparameter tuning, early stopping (where applicable)
  - Tracking experiments and results

- **Workshop 08 — Unsupervised Learning**
  - k‑means, hierarchical clustering, DBSCAN
  - Dimensionality reduction: PCA; visualization with t‑SNE/UMAP

- **Workshop 09 — Neural Networks**
  - MLP fundamentals (TensorFlow/PyTorch)
  - Activations, losses, optimizers
  - Overfitting control: regularization, dropout

- **Workshop 10 — NLP Basics**
  - Text cleaning, tokenization, n‑grams
  - Bag‑of‑Words/TF‑IDF with simple classifiers
  - Intro to embeddings

- **Workshop 11 — Computer Vision Basics**
  - Image preprocessing and augmentation
  - CNN fundamentals and transfer learning demo

- **Workshop 12 — Time Series**
  - Stationarity, decomposition, seasonality
  - ARIMA/Prophet overview; ML forecasting with sliding windows

---

## Repository Structure

```
workshops/
  Workshop-01-Intro-to-Python-NumPy/
  Workshop-02-Data-Preprocessing-EDA/
  Workshop-03-Supervised-Classification/
  Workshop-04-Regression-Models/
  Workshop-05-Model-Evaluation-Metrics/
  Workshop-06-Feature-Engineering/
  Workshop-07-Model-Tuning-CV/
  Workshop-08-Unsupervised-Clustering-DimRed/
  Workshop-09-Neural-Networks/
  Workshop-10-NLP-Basics/
  Workshop-11-Computer-Vision-Basics/
  Workshop-12-Time-Series/
datasets/
reports/
environment/
  requirements.txt  (or environment.yml)
```

Conventions:
- Each workshop may include a top‑level notebook (e.g., `main.ipynb`), a `README.md`, `src/` for scripts, and `outputs/` for generated artifacts.
- Keep datasets small in‑repo; place large files outside the repo and link instructions to download.

---

## Environment Setup (Windows PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r environment/requirements.txt
```

Optional (Anaconda/Miniconda):

```bash
conda create -n aiml python=3.11 -y
conda activate aiml
pip install -r environment/requirements.txt
```

---

## How to Run

Run notebooks:

```bash
pip install jupyterlab
jupyter lab
```

Run a script (example):

```bash
python workshops/Workshop-03-Supervised-Classification/train.py
```

Reproduce a workshop from scratch (example flow):

```bash
cd workshops/Workshop-07-Model-Tuning-CV
python src/preprocess.py
python src/train.py --model random_forest --cv 5
python src/evaluate.py --metrics f1 roc_auc
```

---

## Datasets

- Small sample datasets are stored in `datasets/`.
- For larger datasets, each workshop `README.md` provides links and download instructions.
- If a dataset must not be shared publicly, keep it out of Git (use `.gitignore`) and document the source.

---

## Results and Reports

- Key metrics, charts, and tables are saved under each workshop’s `outputs/` and summarized in `reports/`.
- When relevant, include a brief discussion of findings and limitations.

---

## Reproducibility

- Fixed random seeds where applicable (NumPy, PyTorch/TensorFlow, scikit‑learn).
- Note that some operations (e.g., multithreading, GPU kernels) can be nondeterministic.
- Record exact package versions in `environment/requirements.txt`.

---

## Academic Integrity

- This repository is for learning and assessment submission.
- Cite all external sources (datasets, code snippets, papers, tutorials) within notebooks or `reports/`.
- Do not share restricted coursework materials.

---

## License

Choose one appropriate for your course/institution:

- MIT License (permissive)
- Apache 2.0 (permissive with patent grant)
- All rights reserved (if required for submissions)

Add the chosen license text at the repository root.

---

## Maintainer

- Name: Kavishka Herath (ID: 2548577)
- Module: Artificial Intelligence & Machine Learning (3rd Year)


