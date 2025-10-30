overview content for all workshops
Purpose: Central repository for AI/ML workshops and practicals, including datasets, notebooks, and reports. Each workshop is self-contained and reproducible.
Repo structure
workshops/Workshop-01-Intro-to-Python-NumPy/
workshops/Workshop-02-Data-Preprocessing-EDA/
workshops/Workshop-03-Supervised-Classification/
workshops/Workshop-04-Regression-Models/
workshops/Workshop-05-Model-Evaluation-Metrics/
workshops/Workshop-06-Feature-Engineering/
workshops/Workshop-07-Model-Tuning-CV/
workshops/Workshop-08-Unsupervised-Clustering-DimRed/
workshops/Workshop-09-Neural-Networks/
workshops/Workshop-10-NLP-Basics/
workshops/Workshop-11-Computer-Vision-Basics/
workshops/Workshop-12-Time-Series/
datasets/ (sample or links)
reports/ (summaries, findings)
environment/ (requirements.txt or environment.yml)
Workshops overview
Workshop 01 — Intro to Python & NumPy: Python basics, NumPy arrays, vectorization, plotting with Matplotlib.
Workshop 02 — Data Preprocessing & EDA: Cleaning, handling missing values, outliers, scaling, visual EDA with Pandas/Seaborn.
Workshop 03 — Supervised Classification: Train/test split, k-NN, Logistic Regression, Decision Trees; baseline vs improved models.
Workshop 04 — Regression Models: Linear/Polynomial Regression, regularization (Ridge/Lasso), error analysis.
Workshop 05 — Model Evaluation: Accuracy, precision/recall, F1, ROC-AUC, confusion matrix, regression metrics (MAE/MSE/RMSE).
Workshop 06 — Feature Engineering: Encoding, scaling, binning, text/image features, feature importance.
Workshop 07 — Model Tuning & Cross-Validation: Grid/Random search, k-fold CV, pipelines, avoiding leakage.
Workshop 08 — Unsupervised Learning: k-means, hierarchical clustering, DBSCAN, PCA/t-SNE/UMAP for visualization.
Workshop 09 — Neural Networks: MLPs in PyTorch/TensorFlow, activation functions, loss, optimizers, overfitting control.
Workshop 10 — NLP Basics: Text cleaning, tokenization, TF-IDF, simple classifiers, intro to word embeddings.
Workshop 11 — Computer Vision Basics: Image preprocessing, augmentation, CNN fundamentals, transfer learning demo.
Workshop 12 — Time Series: Stationarity, decomposition, ARIMA/Prophet, sliding windows for ML forecasting.
How to run
Create environment:
    python -m venv .venv    .venv\Scripts\activate    pip install -r environment/requirements.txt    ```  - Launch notebooks:    ```bash    jupyter lab    ```  - Or run scripts:    ```bash    python workshops/Workshop-03-Supervised-Classification/train.py    ```- **Datasets**  - Small samples included in `datasets/`. Larger datasets are linked in each workshop README with download instructions.- **Reproducibility**  - Fixed random seeds where applicable.  - Note any nondeterministic steps (e.g., parallelism, GPU ops).- **Results and reports**  - Key metrics, charts, and conclusions saved to `reports/` and per-workshop output folders.- **Academic integrity**  - This repo is for learning and assessment submission. Cite all external sources and do not share restricted coursework materials.- **License**  - Choose one (e.g., MIT) or mark “All rights reserved” if required by your course.
Launch notebooks:
    jupyter lab
Or run scripts:
    python workshops/Workshop-03-Supervised-Classification/train.py
Datasets
Small samples included in datasets/. Larger datasets are linked in each workshop README with download instructions.
Reproducibility
Fixed random seeds where applicable.
Note any nondeterministic steps (e.g., parallelism, GPU ops).
Results and reports
Key metrics, charts, and conclusions saved to reports/ and per-workshop output folders.
Academic integrity
This repo is for learning and assessment submission. Cite all external sources and do not share restricted coursework materials.
License
Choose one (e.g., MIT) or mark “All rights reserved” if required by your course.
