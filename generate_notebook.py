import json

def make_notebook():
    cells = []

    def md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    md("""
# (*≧ω≦*) OchreML: Fast Classical ML in Rust with PyO3 Bindings (＾▽＾)
### A Complete Tour of Linear Regression and Decision Trees with NumPy, Pandas, and Polars Interop

Welcome to the official demonstration notebook for **OchreML**! (*^▽^*)  
OchreML is a high-performance classical machine learning library with an analytical Rust engine, seamlessly wrapped into a clean, scikit-learn-compatible Python interface using PyO3.

---

### Key Highlights (ﾉ◕ヮ◕)ﾉ*:・ﾟ✧
- **Linear Regression (OLS / Normal Equation):** Exact analytical solution with automated Tikhonov (Ridge) regularization fallback.
- **Decision Tree Classifier:** CART implementation supporting both **Gini Impurity** and **Entropy** splitting criteria.
- **Decision Tree Regressor:** Non-linear regression tree minimizing **Mean Squared Error (MSE / Variance Reduction)**.
- **Seamless Multi-Format Interoperability:** Works natively with standard Python nested lists, **NumPy ndarrays**, **Pandas DataFrames**, and **Polars DataFrames**!
- **Pure Kaomoji Spirit:** Friendly and expressive documentation with zero Unicode emojis (o´∀｀o).
""")

    # Setup & Installation
    md("""
## 1. Installation & Environment Setup (o´∀｀o)

In Kaggle notebooks, you can install OchreML either from PyPI or directly from the official GitHub repository:
""")

    code("""
# Install OchreML (from PyPI or directly from GitHub repository)
# !pip install ochreml
# Or build from source on Kaggle:
# !pip install git+https://github.com/arufadesuwa/ochreml.git

import ochreml
from ochreml import LinearRegression, DecisionTreeClassifier, DecisionTreeRegressor

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

print("OchreML loaded successfully! (*≧ω≦*)")
print("Version:", ochreml.__version__)
""")

    # Linear Regression Section
    md("""
---
## 2. Linear Regression (OLS & Normal Equation) (*^▽^*)

OchreML solves the Ordinary Least Squares problem analytically via the Normal Equation:
$$\\theta = (X^T X)^{-1} X^T y$$

If the design matrix is collinear or ill-conditioned, OchreML's internal linear solver automatically engages Tikhonov regularization $(X^T X + \\lambda I)$ to guarantee numerical stability! (＾▽＾)

Let's test this with a dataset containing noise:
""")

    code("""
# Generate synthetic linear regression data
np.random.seed(42)
n_samples = 60
X_raw = np.linspace(0.0, 10.0, n_samples)
true_slope = 2.5
true_intercept = 5.0
noise = np.random.normal(0.0, 1.8, n_samples)
y_raw = true_slope * X_raw + true_intercept + noise

# Package features into a Pandas DataFrame
df_train = pd.DataFrame({"feature_x": X_raw})
y_train = pd.Series(y_raw, name="target_y")

# Initialize and fit LinearRegression
lr_model = LinearRegression(fit_intercept=True)
lr_model.fit(df_train, y_train)

# Inspect model parameters
print("Fitted Coefficients (coef_):", lr_model.coef_)
print(f"Fitted Intercept (intercept_): {lr_model.intercept_:.4f}")
print(f"R^2 Score on Training Set: {lr_model.score(df_train, y_train):.4f} (*≧ω≦*)")
""")

    md("""
### Visualizing the Regression Fit (o´∀｀o)
Let's plot our data points alongside OchreML's fitted regression line:
""")

    code("""
# Generate dense evaluation grid
x_grid = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
y_preds = lr_model.predict(x_grid)

plt.figure(figsize=(9, 5))
plt.scatter(X_raw, y_raw, color="#3498db", edgecolors="black", alpha=0.8, label="Observed Samples")
plt.plot(x_grid, y_preds, color="#e74c3c", linewidth=2.5, label=f"OchreML OLS Fit (R^2 = {lr_model.score(df_train, y_train):.3f})")
plt.title("OchreML: Linear Regression Fit (＾▽＾)", fontsize=14, fontweight="bold")
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True)
plt.show()
""")

    # Decision Tree Classifier Section
    md("""
---
## 3. Decision Tree Classifier (Gini Impurity vs Entropy) (★ω★)

OchreML implements the CART algorithm for classification. You can customize:
- `criterion`: `"gini"` or `"entropy"`
- `max_depth`: Depth ceiling to prevent overfitting
- `min_samples_split`: Minimum sample count required to split an internal node
- `min_samples_leaf`: Minimum sample count in leaf nodes

Let's create a non-linear 2D classification problem and test with a **Polars DataFrame**! ＼(＾▽＾)／
""")

    code("""
# Generate synthetic 2-class non-linear dataset
np.random.seed(1337)
n_points = 120

# Class 0: inner cluster; Class 1: outer ring/clusters
r0 = np.random.uniform(0.5, 2.0, n_points // 2)
theta0 = np.random.uniform(0, 2 * np.pi, n_points // 2)
x0 = r0 * np.cos(theta0)
y0 = r0 * np.sin(theta0)

r1 = np.random.uniform(3.0, 5.0, n_points // 2)
theta1 = np.random.uniform(0, 2 * np.pi, n_points // 2)
x1 = r1 * np.cos(theta1)
y1 = r1 * np.sin(theta1)

X_cls = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
y_cls = np.array([0.0] * (n_points // 2) + [1.0] * (n_points // 2))

# Construct a Polars DataFrame for training
pl_train = pl.DataFrame({
    "x1": X_cls[:, 0],
    "x2": X_cls[:, 1],
})
pl_target = pl.Series("label", y_cls)

print("Training data shape in Polars:", pl_train.shape, "(o´∀｀o)")
""")

    md("""
### Training Decision Trees & Comparing Criteria (*^▽^*)
Let's train two trees: one using **Gini Impurity** and one using **Entropy**:
""")

    code("""
# Train tree with Gini criterion
tree_gini = DecisionTreeClassifier(criterion="gini", max_depth=4)
tree_gini.fit(pl_train, pl_target)

# Train tree with Entropy criterion
tree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=4)
tree_entropy.fit(pl_train, pl_target)

print(f"Gini Tree Accuracy: {tree_gini.score(pl_train, pl_target) * 100:.2f}% (＾▽＾)")
print(f"Entropy Tree Accuracy: {tree_entropy.score(pl_train, pl_target) * 100:.2f}% (*≧ω≦*)")
print("Identified Classes:", tree_gini.classes_)
""")

    md("""
### Plotting the Decision Boundary Contour (ﾉ◕ヮ◕)ﾉ*:・ﾟ✧
Visualizing how OchreML's orthogonal axis-aligned splits carve the feature space:
""")

    code("""
# Create evaluation grid
x_min, x_max = X_cls[:, 0].min() - 1, X_cls[:, 0].max() + 1
y_min, y_max = X_cls[:, 1].min() - 1, X_cls[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict on grid points using Gini tree
Z_gini = tree_gini.predict(grid_points).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 7))
ax.contourf(xx, yy, Z_gini, alpha=0.35, cmap="coolwarm")
scatter = ax.scatter(X_cls[:, 0], X_cls[:, 1], c=y_cls, cmap="coolwarm", edgecolors="k", s=45)
ax.set_title("OchreML: Decision Tree Boundary Contour (Gini, max_depth=4) (*≧ω≦*)", fontsize=13, fontweight="bold")
ax.set_xlabel("Feature X1")
ax.set_ylabel("Feature X2")
ax.grid(True, linestyle="--", alpha=0.4)
plt.show()
""")

    # Decision Tree Regressor Section
    md("""
---
## 4. Decision Tree Regressor (Non-Linear Fitting) (o´∀｀o)

For continuous target estimation, `DecisionTreeRegressor` minimizes the Mean Squared Error (MSE) / Variance of subsets at every branch.

Let's observe how the tree depth parameter controls the approximation of a non-linear sine wave function:
""")

    code("""
# Generate non-linear 1D sine wave data
np.random.seed(99)
X_sine = np.sort(np.random.uniform(0.0, 2 * np.pi, 80)).reshape(-1, 1)
y_sine = np.sin(X_sine.ravel()) + np.random.normal(0.0, 0.15, 80)

# Train two regressors with different depths
tree_reg_d2 = DecisionTreeRegressor(criterion="squared_error", max_depth=2)
tree_reg_d2.fit(X_sine, y_sine)

tree_reg_d5 = DecisionTreeRegressor(criterion="squared_error", max_depth=5)
tree_reg_d5.fit(X_sine, y_sine)

print(f"Depth 2 Tree R^2 Score: {tree_reg_d2.score(X_sine, y_sine):.4f} (o´∀｀o)")
print(f"Depth 5 Tree R^2 Score: {tree_reg_d5.score(X_sine, y_sine):.4f} (★ω★)")
""")

    md("""
### Visualizing Non-Linear Step Functions (*^▽^*)
""")

    code("""
# Predict on continuous curve
X_eval = np.linspace(0.0, 2 * np.pi, 300).reshape(-1, 1)
y_pred_d2 = tree_reg_d2.predict(X_eval)
y_pred_d5 = tree_reg_d5.predict(X_eval)

plt.figure(figsize=(10, 5))
plt.scatter(X_sine, y_sine, s=30, edgecolor="black", c="navy", alpha=0.7, label="Noisy Data")
plt.plot(X_eval, np.sin(X_eval), color="gray", linestyle="--", label="Ground Truth sin(x)")
plt.plot(X_eval, y_pred_d2, color="#f39c12", linewidth=2.0, label="OchreML Tree (max_depth=2)")
plt.plot(X_eval, y_pred_d5, color="#27ae60", linewidth=2.0, label="OchreML Tree (max_depth=5)")
plt.title("OchreML: Decision Tree Regression Fitting (o´∀｀o)", fontsize=13, fontweight="bold")
plt.xlabel("X")
plt.ylabel("Target y")
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
""")

    # Multi-Format Interoperability Section
    md("""
---
## 5. Universal Multi-Format Interoperability (≧∇≦)/

One of OchreML's core strengths is zero-hassle interoperability. You can train with a **Pandas DataFrame**, predict with a **Polars DataFrame**, evaluate with a **NumPy ndarray**, or pass **standard Python nested lists**! (＾▽＾)
""")

    code("""
# 1. Train on Pandas
p_df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0], "feat2": [2.0, 1.0, 4.0, 3.0]})
p_target = pd.Series([10.0, 20.0, 30.0, 40.0])
model_interop = LinearRegression()
model_interop.fit(p_df, p_target)

# 2. Predict using Polars
pl_query = pl.DataFrame({"feat1": [2.5, 3.5], "feat2": [1.5, 3.5]})
pl_result = model_interop.predict(pl_query)

# 3. Predict using Pure Python Nested Lists
list_query = [[1.0, 2.0], [4.0, 3.0]]
list_result = model_interop.predict(list_query)

# 4. Predict using NumPy ndarray
np_query = np.array([[2.0, 1.0]])
np_result = model_interop.predict(np_query)

print("Predictions from Polars input:", pl_result, "(*≧ω≦*)")
print("Predictions from Python Nested List:", list_result, "(o´∀｀o)")
print("Predictions from NumPy array:", np_result, "(＾▽＾)")
""")

    # Conclusion Section
    md("""
---
## 6. Summary & Conclusion ＼(＾▽＾)／

OchreML bridges native Rust performance with the elegance of modern Python data science workflows:
- **Fast and robust OLS:** Closed-form solution with Tikhonov matrix regularization.
- **Interpretable Decision Trees:** Both classification and regression trees with customizable hyperparameters.
- **Universal input handling:** Works out-of-the-box with List, NumPy, Pandas, and Polars.
- **Zero Unicode emojis:** Clean Kaomoji documentation styling throughout.

### Useful Links:
- **GitHub Repository:** [github.com/arufadesuwa/ochreml](https://github.com/arufadesuwa/ochreml)
- **PyPI Package:** [pypi.org/project/ochreml](https://pypi.org/project/ochreml)

*Thank you for exploring OchreML! Happy machine learning! (*≧ω≦*)*
""")

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open("ochreml_demo_kaggle.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print("Successfully generated ochreml_demo_kaggle.ipynb! (*≧ω≦*)")

if __name__ == "__main__":
    make_notebook()
