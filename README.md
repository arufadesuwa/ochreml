# OchreML (*≧ω≦*)

**Classical Machine Learning Library in Rust with PyO3 Bindings**

Welcome to **OchreML**! (＾▽＾)  
OchreML is a classical Machine Learning library (Linear Regression & Decision Trees) combining Rust's blazingly fast computational performance with Python's intuitive, scikit-learn-compatible API.

---

## Key Features (*^▽^*)

1. **Linear Regression (OLS / Normal Equation)**
   - Exact analytical solution using the Normal Equation $(X^T X)^{-1} X^T y$.
   - Automated Tikhonov (Ridge) regularization fallback when design matrices are ill-conditioned or singular.
   - Configurable `fit_intercept` parameter for calculating bias.

2. **Decision Tree Classifier**
   - CART classification trees supporting both **Gini Impurity** and **Entropy** criteria.
   - Flexible hyperparameters: `max_depth`, `min_samples_split`, and `min_samples_leaf`.

3. **Decision Tree Regressor**
   - Non-linear regression trees based on **Mean Squared Error (MSE / Variance Reduction)** minimization.
   - Robust continuous target estimation.

4. **Multi-Format Data Interoperability** (o´∀｀o)
   - Seamlessly accepts and processes:
     - **Python Nested Lists** (`list[list[float]]`)
     - **NumPy ndarrays** (`numpy.ndarray`)
     - **Pandas DataFrames & Series** (`pandas.DataFrame`, `pandas.Series`)
     - **Polars DataFrames & Series** (`polars.DataFrame`, `polars.Series`)

5. **Scikit-Learn Compatible API** (＾▽＾)
   - Standard estimator interface: `fit(X, y)`, `predict(X)`, and `score(X, y)`.
   - Inspection attributes: `coef_`, `intercept_`, `classes_`, and `n_features_in_`.

---

## Build & Installation (o´∀｀o)

Ensure Rust (cargo) and your Python virtual environment are activated:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Compile and install locally with Maturin
maturin develop --release
```

---

## Quickstart Examples (＾▽＾)

### 1. Linear Regression with NumPy & Pandas

```python
import numpy as np
import pandas as pd
from ochreml import LinearRegression

# Prepare dataset with Pandas DataFrame
df = pd.DataFrame({
    "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
    "x2": [2.0, 1.0, 4.0, 3.0, 5.0],
})
# Target relationship: y = 2*x1 + 3*x2 + 5
y = np.array([13.0, 12.0, 23.0, 22.0, 30.0])

model = LinearRegression(fit_intercept=True)
model.fit(df, y)

print("Coefficients:", model.coef_)       # approx [2.0, 3.0]
print("Intercept:", model.intercept_)     # approx 5.0
print("R^2 Score:", model.score(df, y))   # 1.0

# Predict on new samples
X_new = pd.DataFrame({"x1": [6.0], "x2": [2.0]})
print("Prediction:", model.predict(X_new)) # [23.0]
```

### 2. Decision Tree Classifier with Polars

```python
import polars as pl
from ochreml import DecisionTreeClassifier

# Prepare dataset with Polars
df_polars = pl.DataFrame({
    "study_hours": [1.0, 2.0, 2.5, 5.0, 6.0, 7.5, 8.0, 9.0],
    "attendance": [50.0, 60.0, 40.0, 80.0, 85.0, 90.0, 95.0, 100.0],
})
y_polars = pl.Series("passed", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
clf.fit(df_polars, y_polars)

print("Classes discovered:", clf.classes_)
print("Training accuracy:", clf.score(df_polars, y_polars))

# Predict on new samples
test_data = pl.DataFrame({"study_hours": [1.5, 7.0], "attendance": [55.0, 88.0]})
print("Predictions:", clf.predict(test_data)) # [0.0, 1.0]
```

### 3. Decision Tree Regressor with Standard Python Lists

```python
from ochreml import DecisionTreeRegressor

# Dataset using standard Python nested lists
X = [[100.0], [300.0], [500.0], [1500.0], [1800.0], [2200.0]]
y = [31.0, 30.0, 29.5, 17.0, 16.0, 15.0]

reg = DecisionTreeRegressor(criterion="squared_error", max_depth=2)
reg.fit(X, y)

print("R^2 Score:", reg.score(X, y))
print("Prediction [200.0m]:", reg.predict([[200.0]]))   # [31.0]
print("Prediction [2000.0m]:", reg.predict([[2000.0]])) # [15.5]
```

---

## Testing Suite (★ω★)

```bash
# Run internal Rust unit tests
cargo test

# Run Python integration tests (Lists, NumPy, Pandas, Polars)
pytest
```
