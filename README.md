🧠 AI/ML Workshops Repository
Purpose

A central repository for all AI/ML workshops and practicals — including datasets, notebooks, and reports.
Each workshop is self-contained, reproducible, and designed for hands-on learning and academic use.


Workshops Overview

No.	Workshop Title	Key Topics
01	Intro to Python & NumPy	Python basics, NumPy arrays, vectorization, plotting with Matplotlib
02	Data Preprocessing & EDA	Cleaning, missing values, outliers, scaling, visual EDA with Pandas/Seaborn
03	Supervised Classification	Train/test split, k-NN, Logistic Regression, Decision Trees, baseline vs improved models
04	Regression Models	Linear & Polynomial Regression, regularization (Ridge/Lasso), error analysis
05	Model Evaluation Metrics	Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, MAE/MSE/RMSE
06	Feature Engineering	Encoding, scaling, binning, text/image features, feature importance
07	Model Tuning & Cross-Validation	Grid/Random search, k-fold CV, pipelines, data leakage prevention
08	Unsupervised Learning	k-means, hierarchical, DBSCAN, PCA/t-SNE/UMAP for visualization
09	Neural Networks	MLPs with PyTorch/TensorFlow, activations, loss, optimizers, overfitting control
10	NLP Basics	Text cleaning, tokenization, TF-IDF, simple classifiers, intro to embeddings
11	Computer Vision Basics	Image preprocessing, augmentation, CNN fundamentals, transfer learning demo
12	Time Series Analysis	Stationarity, decomposition, ARIMA/Prophet, ML-based forecasting

How to Run
1️⃣ Create Environment
python -m venv .venv
.venv\Scripts\activate    # (Windows)
# source .venv/bin/activate   # (Mac/Linux)
pip install -r environment/requirements.txt

2️⃣ Launch Notebooks
jupyter lab

3️⃣ Run Scripts (Example)
python workshops/Workshop-03-Supervised-Classification/train.py

Datasets

Small sample datasets are included in datasets/.

Larger datasets are linked in each workshop’s README with download instructions.

Reproducibility

Fixed random seeds where applicable.

Noted any nondeterministic steps (e.g., GPU operations, parallel processing).

Results & Reports

Key metrics, charts, and conclusions are saved under:

reports/

Per-workshop output/ folders.

Academic Integrity

This repository is for learning and assessment submission only.
Cite all external sources properly and do not share restricted coursework materials.

License

Choose one of the following:

MIT License — open for reuse and learning.

All Rights Reserved — if required by your course.

✅ Maintained as part of ongoing AI/ML coursework to support learning, reproducibility, and structured skill development.
