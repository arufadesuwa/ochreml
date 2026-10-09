# OchreML (*≧ω≦*)

**Classical Machine Learning Library in Rust with PyO3 Bindings**

Welcome to **OchreML**! (＾▽＾)  
OchreML is a high-performance classical Machine Learning library written in Rust with seamless Python bindings via PyO3. Designed with scikit-learn API compatibility, OchreML delivers pure native execution speed, zero unnecessary memory copies, and flexible multi-format data interoperability.

---

## Key Features (*^▽^*)

1. **High-Performance Rust Kernels**
   - $O(D \cdot N \log N)$ CART split search with sliding histogram impurity evaluations.
   - Closed-form Normal Equation solver with symmetric Gram matrix accumulation and automated Tikhonov (Ridge) regularization fallback.
   - Outperforms standard Cython/C implementations on Decision Tree training and prediction.

2. **Universal Data Interoperability** (o´∀｀o)
   - Accepts and processes data seamlessly across:
     - **Python Nested Lists** (`list[list[float]]`)
     - **NumPy ndarrays** (`numpy.ndarray`)
     - **Pandas DataFrames & Series** (`pandas.DataFrame`, `pandas.Series`)
     - **Polars DataFrames & Series** (`polars.DataFrame`, `polars.Series`)

3. **Scikit-Learn Compatible API** (＾▽＾)
   - Standard estimator methods: `fit(X, y)`, `predict(X)`, and `score(X, y)`.
   - Inspection attributes: `coef_`, `intercept_`, `classes_`, and `n_features_in_`.

---

## Model Catalog & Roadmap (＾▽＾)

### Currently Available Models

| Model | Module | Primary Algorithm & Characteristics |
| :--- | :--- | :--- |
| **`LinearRegression`** | `ochreml.LinearRegression` | Analytical Ordinary Least Squares (OLS) via Normal Equation $(X^T X)^{-1} X^T y$ with automated Tikhonov Ridge fallback. |
| **`DecisionTreeClassifier`** | `ochreml.DecisionTreeClassifier` | Optimized CART classification tree supporting both **Gini Impurity** and **Entropy** splitting criteria with sliding histograms. |
| **`DecisionTreeRegressor`** | `ochreml.DecisionTreeRegressor` | Fast non-linear regression tree minimizing **Mean Squared Error (MSE / Variance Reduction)** with $O(1)$ running sum variance updates. |

### Upcoming Models (Roadmap) (*^▽^*)

- **Logistic Regression**: Binary and multinomial classification via L-BFGS / Newton-Raphson solvers.
- **Random Forest**: Ensemble bagging for both Classifier and Regressor with multi-threaded tree building.
- **K-Nearest Neighbors (KNN)**: Fast KD-Tree / Ball-Tree spatial search for classification and regression.
- **Support Vector Machines (SVM)**: Sequential Minimal Optimization (SMO) kernel classifiers.
- **Naive Bayes**: Gaussian and Multinomial probabilistic classifiers.
- **Principal Component Analysis (PCA)**: Truncated SVD / covariance eigendecomposition for dimensionality reduction.
- **K-Means Clustering**: Lloyd's algorithm with K-Means++ initialization.

---

## Performance Benchmarks (*≧ω≦*)

Comprehensive benchmark comparing OchreML's pure Rust engine against Scikit-Learn (Cython backend).

### 1. Decision Tree Classifier (`criterion="gini"`, `max_depth=5`)
*Dataset: 2,500 samples (2,000 train, 500 test), 8 features*

| Metric | OchreML (Rust) | Scikit-Learn | Ratio / Parity |
| :--- | :--- | :--- | :--- |
| **Fit Time** | **5.55 ms** | 9.23 ms | **1.66x FASTER** (*≧ω≦*) |
| **Predict Time** | **0.27 ms** | 0.38 ms | **1.40x FASTER** |
| **Test Accuracy** | **92.40 %** | 92.40 % | Identical Parity |

### 2. Decision Tree Regressor (`criterion="squared_error"`, `max_depth=5`)
*Dataset: 2,500 samples (2,000 train, 500 test), 6 features*

| Metric | OchreML (Rust) | Scikit-Learn | Ratio / Parity |
| :--- | :--- | :--- | :--- |
| **Fit Time** | **3.72 ms** | 5.79 ms | **1.55x FASTER** (*^▽^*) |
| **Predict Time** | **0.25 ms** | 0.34 ms | **1.36x FASTER** |
| **Test $R^2$ Score** | **0.870028** | 0.870028 | Identical Parity |

### 3. Linear Regression (OLS Normal Equation)
*Dataset: 20,000 samples (16,000 train, 4,000 test), 10 features*

| Metric | OchreML (Rust) | Scikit-Learn | Ratio / Parity |
| :--- | :--- | :--- | :--- |
| **Fit Time** | 17.05 ms | 9.11 ms | Analytical Normal Equation |
| **Predict Time** | 2.60 ms | 0.44 ms | Zero-copy row slicing |
| **Test $R^2$ Score** | **0.997970** | 0.997970 | Max diff: $2.86 \times 10^{-14}$ |

---

## Installation & Build (o´∀｀o)

### From PyPI

```bash
pip install ochreml
```

### From Source

```bash
# Clone the repository
git clone https://github.com/arufadesuwa/ochreml.git
cd ochreml

# Build and install locally with Maturin
maturin develop --release
```

---

## Testing Suite (o´∀｀o)

```bash
# Run Rust unit tests
cargo test

# Run Python integration test suite (Lists, NumPy, Pandas, Polars)
pytest
```
