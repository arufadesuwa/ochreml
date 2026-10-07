# Demo Penggunaan OchreML (＾▽＾)
# Integrasi Python List, NumPy ndarray, Pandas DataFrame, dan Polars DataFrame.

import numpy as np
import pandas as pd
import polars as pl

from ochreml import LinearRegression, DecisionTreeClassifier, DecisionTreeRegressor


def banner(title: str, kaomoji: str):
    print("\n" + "=" * 65)
    print(f" {kaomoji}  {title}")
    print("=" * 65)


def demo_linear_regression():
    banner("1. LINEAR REGRESSION (Pandas & NumPy)", "(*^▽^*)")

    # Fitur: [area_sqm, rooms]
    df_features = pd.DataFrame({
        "area_sqm": [36.0, 45.0, 60.0, 72.0, 90.0, 120.0],
        "rooms": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
    })
    # Target: 10 * area + 25 * rooms + 50
    y = np.array([435.0, 550.0, 700.0, 845.0, 1025.0, 1350.0])

    print("Training Features (Pandas DataFrame):")
    print(df_features)

    model = LinearRegression(fit_intercept=True)
    print("\nTraining Linear Regression with Normal Equation...")
    model.fit(df_features, y)

    print("\nTraining Results (＾▽＾):")
    print(f" - Coefficients (coef_): {model.coef_}")
    print(f" - Intercept (intercept_): {model.intercept_:.2f}")
    print(f" - R^2 Score: {model.score(df_features, y):.4f}")

    # Prediksi menggunakan NumPy array
    x_new = np.array([[50.0, 2.0], [100.0, 3.0]])
    predictions = model.predict(x_new)
    print("\nPredictions for new houses [50 sqm, 2 rooms] and [100 sqm, 3 rooms] (*≧ω≦*):")
    for i, p in enumerate(predictions):
        print(f"   House {i + 1}: ${p:.2f}k")


def demo_decision_tree_classifier():
    banner("2. DECISION TREE CLASSIFIER (Polars & Gini Impurity)", "(★ω★)")

    # Data menggunakan Polars DataFrame dan Series
    df_students = pl.DataFrame({
        "study_hours": [1.0, 2.0, 2.5, 5.0, 6.0, 7.5, 8.0, 9.0],
        "attendance_pct": [50.0, 60.0, 40.0, 80.0, 85.0, 90.0, 95.0, 100.0],
    })
    y_students = pl.Series("passed", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    print("Student Dataset (Polars DataFrame):")
    print(df_students)

    clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
    print("\nFitting Decision Tree Classifier...")
    clf.fit(df_students, y_students)

    print("\nEvaluation Results (＾▽＾):")
    print(f" - Registered Classes: {clf.classes_}")
    print(f" - Training Accuracy: {clf.score(df_students, y_students) * 100:.1f}%")

    # Prediksi menggunakan nested list Python standar
    test_data = [[1.5, 55.0], [7.0, 88.0]]
    preds = clf.predict(test_data)
    status = ["Failed (´-ω-`)", "Passed (*≧ω≦*)"]
    print("\nPredictions for New Students (o´∀｀o):")
    for idx, p in enumerate(preds):
        print(f"   Student {idx + 1} ({test_data[idx]}): {status[int(p)]}")


def demo_decision_tree_regressor():
    banner("3. DECISION TREE REGRESSOR (Nested List & MSE)", "(o´∀｀o)")

    # Data menggunakan nested list Python standar
    X = [[100.0], [300.0], [500.0], [1500.0], [1800.0], [2200.0]]
    y = [31.0, 30.0, 29.5, 17.0, 16.0, 15.0]

    print("Training Data (Python Nested List):")
    for r_x, r_y in zip(X, y):
        print(f" - Elevation: {r_x[0]} m -> Temperature: {r_y} C")

    reg = DecisionTreeRegressor(criterion="squared_error", max_depth=2)
    print("\nFitting Decision Tree Regressor...")
    reg.fit(X, y)

    print("\nEvaluation Results (★ω★):")
    print(f" - R^2 Score: {reg.score(X, y):.4f}")

    # Prediksi elevasi baru
    test_elevations = [[200.0], [1000.0], [2000.0]]
    temp_preds = reg.predict(test_elevations)
    print("\nEstimated Temperatures for New Elevations (*^▽^*):")
    for h, s in zip(test_elevations, temp_preds):
        print(f"   Elevation {h[0]} m -> Estimated Temp: {s:.2f} C")


if __name__ == "__main__":
    print("\n" + "#" * 65)
    print(" (*≧ω≦*) OCHREML OFFICIAL DEMO: LINEAR REGRESSION & DECISION TREE (*≧ω≦*)")
    print("       Classical Machine Learning in Rust with PyO3 Bindings")
    print("#" * 65)

    demo_linear_regression()
    demo_decision_tree_classifier()
    demo_decision_tree_regressor()

    print("\n" + "=" * 65)
    print(" All demonstrations completed successfully! (＾▽＾)")
    print("=" * 65 + "\n")
