# OchreML (*≧ω≦*)

**Classical Machine Learning Library in Rust with PyO3 Bindings**

Welcome to **OchreML**! (＾▽＾)  
OchreML is a high-performance classical Machine Learning library written in Rust with seamless Python bindings via PyO3. Designed with scikit-learn API compatibility, OchreML delivers pure native execution speed, zero unnecessary memory copies, and flexible multi-format data interoperability.

---

## Key Features (*^▽^*)

1. **Hardware Acceleration (CPU, GPU, Multi-GPU, & TPU)** (*≧ω≦*)
   - Native device engine with smart automatic hardware detection (`device="auto"`): auto-prioritizes TPU -> Multi-GPU -> GPU -> CPU.
   - Data-parallel row partitioning across multiple GPUs ($\sum_k X_k^T X_k$) with all-reduce aggregation.
   - Block-tiled systolic matrix accumulation designed for Tensor Processing Unit (TPU) MXUs.
   - Diagnostics and device routing utilities (`ochreml.get_available_devices()`, `ochreml.get_device_info()`, `ochreml.set_default_device()`).

2. **High-Performance Rust Kernels**
   - $O(D \cdot N \log N)$ CART split search with sliding histogram impurity evaluations.
   - Closed-form Normal Equation solver with symmetric Gram matrix accumulation and automated Tikhonov (Ridge) regularization fallback.
   - Outperforms standard Cython/C implementations on Decision Tree training and prediction.

3. **Universal Data Interoperability** (o´∀｀o)
   - Accepts and processes data seamlessly across:
     - **Python Nested Lists** (`list[list[float]]`)
     - **NumPy ndarrays** (`numpy.ndarray`)
     - **Pandas DataFrames & Series** (`pandas.DataFrame`, `pandas.Series`)
     - **Polars DataFrames & Series** (`polars.DataFrame`, `polars.Series`)

4. **Scikit-Learn Compatible API** (＾▽＾)
   - Standard estimator methods: `fit(X, y)`, `predict(X)`, and `score(X, y)`.
   - Inspection attributes: `coef_`, `intercept_`, `classes_`, `device_`, and `n_features_in_`.

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

Comprehensive benchmark comparing OchreML's pure Rust engine (with Zero-Copy Python Buffer Protocol, SIMD, and Rayon multi-threading) against Scikit-Learn (Cython backend).

### 1. Linear Regression (OLS Normal Equation & Cholesky Solver)
*Dataset: 20,000 samples (16,000 train, 4,000 test), 10 features*

| Metric | OchreML (Rust) | Scikit-Learn | Ratio / Parity |
| :--- | :--- | :--- | :--- |
| **Fit Time** | **2.62 ms** | 7.99 ms | **3.05x FASTER** (*≧ω≦*) |
| **Predict Time** | **0.60 ms** | 0.33 ms | Zero-copy vector dot product |
| **Test $R^2$ Score** | **0.997970** | 0.997970 | Max coef diff: $7.11 \times 10^{-15}$ |

### 2. Decision Tree Classifier (`criterion="gini"`, `max_depth=5`)
*Dataset: 2,500 samples (2,000 train, 500 test), 8 features*

| Metric | OchreML (Rust) | Scikit-Learn | Ratio / Parity |
| :--- | :--- | :--- | :--- |
| **Fit Time** | **4.17 ms** | 9.69 ms | **2.32x FASTER** (*^▽^*) |
| **Predict Time** | **0.10 ms** | 0.40 ms | **4.00x FASTER** |
| **Test Accuracy** | **92.40 %** | 92.40 % | Identical Parity |

### 3. Decision Tree Regressor (`criterion="squared_error"`, `max_depth=5`)
*Dataset: 2,500 samples (2,000 train, 500 test), 6 features*

| Metric | OchreML (Rust) | Scikit-Learn | Ratio / Parity |
| :--- | :--- | :--- | :--- |
| **Fit Time** | **3.13 ms** | 6.72 ms | **2.15x FASTER** (o´∀｀o) |
| **Predict Time** | **0.09 ms** | 0.34 ms | **3.77x FASTER** |
| **Test $R^2$ Score** | **0.870028** | 0.870028 | Identical Parity |

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
