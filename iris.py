# ============================================================
# IRIS CLASSIFICATION PROJECT
# Supervised Learning - Classification
#
# Models:
# 1. Logistic Regression
# 2. K-Nearest Neighbors (KNN)
# 3. Decision Tree Classifier
#
# Evaluation:
# - Accuracy
# - Precision
# - Recall
# - F1-Score
# - Confusion Matrix
#
# KNN Experiment:
# k = 1, 3, 5, 7, 9
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
df=pd.read_csv("iris.csv")

# ============================================================
# 2. LOAD DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

# Create DataFrame
df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["target"] = y

# Add species names for easier understanding
df["species"] = df["target"].map(
    {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }
)

print("\n" + "=" * 60)
print("IRIS DATASET")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. BASIC DATA EXPLORATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["species"].value_counts())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

# -----------------------------
# 4.1 Feature Histograms
# -----------------------------

plt.figure(figsize=(12, 8))

df[iris.feature_names].hist(
    figsize=(12, 8),
    bins=15
)

plt.suptitle(
    "Iris Feature Distributions",
    fontsize=16
)

plt.tight_layout()
plt.show()


# -----------------------------
# 4.2 Class Distribution
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="species"
)

plt.title("Iris Class Distribution")
plt.xlabel("Species")
plt.ylabel("Count")

plt.show()


# -----------------------------
# 4.3 Pairplot
# -----------------------------

sns.pairplot(
    df,
    hue="species",
    diag_kind="hist"
)

plt.suptitle(
    "Iris Feature Relationships",
    y=1.02
)

plt.show()


# -----------------------------
# 4.4 Correlation Heatmap
# -----------------------------

plt.figure(figsize=(8, 6))

correlation = df[iris.feature_names].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.show()


# ============================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================

X = df[iris.feature_names]
y = df["target"]


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=200
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test_scaled
)


# ============================================================
# 9. K-NEAREST NEIGHBORS
# ============================================================

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train_scaled,
    y_train
)

knn_pred = knn_model.predict(
    X_test_scaled
)


# ============================================================
# 10. DECISION TREE
# ============================================================

# Decision Trees do not require feature scaling,
# so we use the original X_train and X_test.

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_pred = tree_model.predict(
    X_test
)


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


# ============================================================
# 12. EVALUATE ALL MODELS
# ============================================================

results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_pred
    )
)

results.append(
    evaluate_model(
        "KNN",
        y_test,
        knn_pred
    )
)

results.append(
    evaluate_model(
        "Decision Tree",
        y_test,
        tree_pred
    )
)


# ============================================================
# 13. MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 14. CLASSIFICATION REPORTS
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        logistic_pred,
        target_names=iris.target_names
    )
)


print("\n" + "=" * 60)
print("KNN REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        knn_pred,
        target_names=iris.target_names
    )
)


print("\n" + "=" * 60)
print("DECISION TREE REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        tree_pred,
        target_names=iris.target_names
    )
)


# ============================================================
# 15. CONFUSION MATRIX FUNCTION
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=iris.target_names
    )

    display.plot()

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.show()


# ============================================================
# 16. CONFUSION MATRICES
# ============================================================

plot_confusion_matrix(
    y_test,
    logistic_pred,
    "Logistic Regression"
)

plot_confusion_matrix(
    y_test,
    knn_pred,
    "KNN"
)

plot_confusion_matrix(
    y_test,
    tree_pred,
    "Decision Tree"
)


# ============================================================
# 17. KNN HYPERPARAMETER EXPERIMENT
# ============================================================

k_values = [1, 3, 5, 7, 9]

knn_accuracies = []

print("\n" + "=" * 60)
print("KNN HYPERPARAMETER EXPERIMENT")
print("=" * 60)

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    knn_accuracies.append(
        accuracy
    )

    print(
        f"k = {k} --> Accuracy = {accuracy:.4f}"
    )


# ============================================================
# 18. K VALUE VS ACCURACY PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    knn_accuracies,
    marker="o"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy")

plt.title(
    "K Value vs KNN Accuracy"
)

plt.xticks(k_values)

plt.grid(True)

plt.show()


# ============================================================
# 19. KNN EXPERIMENT TABLE
# ============================================================

knn_experiment_df = pd.DataFrame(
    {
        "K Value": k_values,
        "Accuracy": knn_accuracies
    }
)

print("\n" + "=" * 60)
print("KNN EXPERIMENT RESULTS")
print("=" * 60)

print(
    knn_experiment_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 20. FIND BEST K
# ============================================================

best_k_index = knn_accuracies.index(
    max(knn_accuracies)
)

best_k = k_values[best_k_index]

best_k_accuracy = knn_accuracies[best_k_index]

print("\nBest K:", best_k)
print(
    f"Best K Accuracy: {best_k_accuracy:.4f}"
)


# ============================================================
# 21. FINAL MODEL COMPARISON VISUALIZATION
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

comparison_plot = results_df.set_index(
    "Model"
)[metrics]

comparison_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Model Performance Comparison"
)

plt.ylabel("Score")

plt.ylim(0, 1.1)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Metrics"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print("\nFinal Model Results:")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nKNN K Values Tested:")
print(k_values)

print(
    f"\nBest K value: {best_k}"
)

print(
    f"Best K accuracy: {best_k_accuracy:.4f}"
)

print("\nAll tasks completed successfully.")