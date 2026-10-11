import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)


# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = Path("customer.csv")
CLEAN_FILE = Path("customer_fixed.csv")
GRAPH_DIR = Path("model_graphs")
SHOW_PLOTS = True  # Set to False if you only want PNG files saved.
RANDOM_STATE = 42


# ============================================================
# STEP 1 - REPAIR MALFORMED CSV ROWS
# ============================================================
def repair_csv(input_file=INPUT_FILE, output_file=CLEAN_FILE):
    print("=" * 60)
    print("CHECKING AND REPAIRING CSV")
    print("=" * 60)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find {input_file.resolve()}. Put customer.csv in the same folder as this script."
        )

    with input_file.open("r", encoding="utf-8-sig", newline="") as infile:
        rows = list(csv.reader(infile))

    if not rows:
        raise ValueError(f"{input_file} is empty.")

    expected_columns = len(rows[0])
    print(f"Expected columns: {expected_columns}")

    clean_rows = [rows[0]]  # Always preserve the header.
    malformed_count = 0

    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) == expected_columns:
            clean_rows.append(row)
        else:
            malformed_count += 1
            print(
                f"Skipping malformed line {line_number}: {len(row)} columns"
            )

    with output_file.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(clean_rows)

    print(f"\nOriginal data rows: {max(0, len(rows) - 1)}")
    print(f"Clean data rows:    {max(0, len(clean_rows) - 1)}")
    print(f"Malformed rows skipped: {malformed_count}")
    print(f"Created: {output_file}")


# ============================================================
# STEP 2 - LOAD DATASET
# ============================================================
def load_data():
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(CLEAN_FILE)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    return df


# ============================================================
# STEP 3 - EXPLORE DATA
# ============================================================
def explore_data(df):
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset shape:")
    print(df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDuplicate rows:")
    print(df.duplicated().sum())
    print("\nStatistical summary:")
    print(df.describe(include="all"))


# ============================================================
# STEP 4 - CLEAN DATA
# ============================================================
def clean_data(df):
    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)

    df = df.copy()
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Duplicate rows removed: {before - len(df)}")

    if "TotalCharges" in df.columns:
        # Spaces in TotalCharges become NaN and are removed below.
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )

    missing_before = int(df.isna().sum().sum())
    df = df.dropna().copy()
    missing_after = int(df.isna().sum().sum())

    print(f"Missing values before cleaning: {missing_before}")
    print(f"Missing values after cleaning:  {missing_after}")
    print(f"Final dataset shape: {df.shape}")
    return df


# ============================================================
# STEP 5 - PREPARE FEATURES AND TARGET
# ============================================================
def prepare_target(df):
    print("\n" + "=" * 60)
    print("PREPARING FEATURES AND TARGET")
    print("=" * 60)

    # Find the target without depending on capitalization.
    columns_by_lower = {str(c).strip().lower(): c for c in df.columns}
    possible_targets = ["churn", "exited", "exitedcustomer", "customerstatus"]
    target_column = next(
        (columns_by_lower[name] for name in possible_targets if name in columns_by_lower),
        None,
    )
    if target_column is None:
        raise ValueError(
            "Target column was not found. Expected Churn or a supported alternative."
        )

    print(f"Target column: {target_column}")
    target_mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "churned": 1,
        "stayed": 0,
        "1": 1,
        "0": 0,
    }
    y = df[target_column].astype(str).str.strip().str.lower().map(target_mapping)

    valid_rows = y.notna()
    invalid_count = int((~valid_rows).sum())
    if invalid_count:
        print(f"Removing rows with unrecognized target values: {invalid_count}")

    X = df.drop(columns=[target_column]).loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    # IDs are identifiers, not meaningful customer-behavior features.
    id_columns = [
        col for col in X.columns
        if str(col).strip().lower() in {"customerid", "customer_id"}
    ]
    if id_columns:
        X = X.drop(columns=id_columns)
        print(f"Excluded identifier column(s): {id_columns}")

    print("\nTarget distribution (0 = No Churn, 1 = Churn):")
    print(y.value_counts().sort_index())

    if y.nunique() != 2:
        raise ValueError("The target must contain both classes: No Churn and Churn.")

    return X, y


# ============================================================
# STEP 6 - PREPROCESSING
# ============================================================
def create_preprocessor(X):
    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    print("\n" + "=" * 60)
    print("FEATURE TYPES")
    print("=" * 60)
    print("Numerical columns:", numerical_columns)
    print("Categorical columns:", categorical_columns)

    transformers = []
    if numerical_columns:
        transformers.append(("num", StandardScaler(), numerical_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ============================================================
# STEP 7 - TRAIN / TEST SPLIT
# ============================================================
def split_data(X, y):
    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 8 - CREATE MODELS
# ============================================================
def create_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=10
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=10,
            n_jobs=-1,
        ),
        # Faster than an RBF SVC on this dataset. No probability=True overhead.
        "SVM (Linear)": LinearSVC(
            random_state=RANDOM_STATE, max_iter=10000, class_weight="balanced"
        ),
    }


# ============================================================
# STEP 9 - TRAIN MODELS
# ============================================================
def train_models(models, preprocessor, X_train, y_train):
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining: {name} ...", flush=True)
        pipeline = Pipeline(
            steps=[("preprocessor", clone(preprocessor)), ("model", model)]
        )
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        print(f"{name} trained successfully.", flush=True)

    return trained_models


# ============================================================
# STEP 10 - EVALUATE MODELS
# ============================================================
def evaluate_models(trained_models, X_test, y_test):
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    results = []
    predictions = {}
    score_values = {}

    for name, model in trained_models.items():
        print(f"\nEvaluating: {name} ...", flush=True)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            }
        )

        # ROC uses probabilities where available, otherwise the model's decision score.
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
            score_values[name] = scores
        elif hasattr(model, "decision_function"):
            score_values[name] = model.decision_function(X_test)

        print(f"\n{name}")
        print("-" * 40)
        print(classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0,
        ))

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)
    print("Results saved to model_results.csv")
    return results_df, predictions, score_values


# ============================================================
# STEP 11 - MODEL COMPARISON GRAPH
# ============================================================
def plot_model_comparison(results_df):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    ax = results_df.set_index("Model")[metrics].plot(
        kind="bar", figsize=(12, 6), width=0.82
    )
    ax.set_title("Telco Customer Churn - Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Metric")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = GRAPH_DIR / "model_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved graph: {path}")

    # Separate accuracy graph for quick comparison.
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = results_df.sort_values("Accuracy", ascending=True)
    ax.barh(ordered["Model"], ordered["Accuracy"])
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for y_pos, value in enumerate(ordered["Accuracy"]):
        ax.text(min(value + 0.01, 0.92), y_pos, f"{value:.3f}", va="center")
    plt.tight_layout()
    path = GRAPH_DIR / "accuracy_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved graph: {path}")


# ============================================================
# STEP 12 - CONFUSION MATRICES GRAPH
# ============================================================
def plot_confusion_matrices(predictions, y_test):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    count = len(predictions)
    cols = 3
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4.3))
    axes = np.array(axes).reshape(-1)

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for ax in axes[count:]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices - All Models", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = GRAPH_DIR / "confusion_matrices.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved graph: {path}")

    # Save each matrix separately too.
    for name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig_one, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig_one.tight_layout()
        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        one_path = GRAPH_DIR / f"{filename}_confusion_matrix.png"
        fig_one.savefig(one_path, dpi=200, bbox_inches="tight")
        plt.close(fig_one)


# ============================================================
# STEP 13 - ROC CURVE GRAPH
# ============================================================
def plot_roc_curves(score_values, y_test):
    if not score_values:
        print("ROC graph skipped: no model scores were available.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    for name, scores in score_values.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_value = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_title("ROC Curves - Model Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = GRAPH_DIR / "roc_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved graph: {path}")


# ============================================================
# STEP 14 - RANDOM FOREST FEATURE IMPORTANCE GRAPH
# ============================================================
def plot_random_forest_feature_importance(trained_models):
    if "Random Forest" not in trained_models:
        return

    pipeline = trained_models["Random Forest"]
    forest = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        print("Feature-importance graph skipped: feature names unavailable.")
        return

    importances = forest.feature_importances_
    if len(feature_names) != len(importances):
        print("Feature-importance graph skipped: feature-name count mismatch.")
        return

    top_n = min(15, len(importances))
    indices = np.argsort(importances)[-top_n:]
    names = np.asarray(feature_names)[indices]
    values = importances[indices]
    # Shorter labels are easier to read in the graph.
    names = [str(name).replace("num__", "").replace("cat__", "") for name in names]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names, values)
    ax.set_title(f"Random Forest - Top {top_n} Feature Importances")
    ax.set_xlabel("Feature importance")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = GRAPH_DIR / "random_forest_feature_importance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved graph: {path}")


# ============================================================
# STEP 15 - MAIN
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("TELCO CUSTOMER CHURN MACHINE LEARNING")
    print("=" * 60)

    repair_csv()
    df = load_data()
    explore_data(df)
    df = clean_data(df)

    X, y = prepare_target(df)
    preprocessor = create_preprocessor(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = create_models()
    trained_models = train_models(models, preprocessor, X_train, y_train)

    results_df, predictions, score_values = evaluate_models(
        trained_models, X_test, y_test
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(results_df)
    plot_confusion_matrices(predictions, y_test)
    plot_roc_curves(score_values, y_test)
    plot_random_forest_feature_importance(trained_models)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED")
    print("=" * 60)
    print(f"Model metrics: {Path('model_results.csv').resolve()}")
    print(f"All graphs saved in: {GRAPH_DIR.resolve()}")
    print("Close the graph windows to finish the program.")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
