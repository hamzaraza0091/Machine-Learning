#!/usr/bin/env python3
# ============================================================
# HEART DISEASE PREDICTION PIPELINE
# ============================================================
# This script loads a heart disease dataset (CSV), cleans it, 
# preprocesses features, trains multiple classifiers, and evaluates them.
# It prints metrics and shows confusion matrix plots.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# STEP 1: Load Dataset
# ============================================================
print("="*60)
print("HEART DISEASE DATASET ANALYSIS")
print("="*60)

# Use 'heart.csv' by default or update the filename below if different
data_file = "heart.csv"
df = pd.read_csv("heart.csv")

print("\nLoaded '{}' (shape: {})".format(data_file, df.shape))
print("Column names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# STEP 2: Data Inspection
# ============================================================
print("\n" + "="*60)
print("DATA INSPECTION")
print("="*60)

# Check for missing values
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "None")

# Check for duplicate rows
dup_count = df.duplicated().sum()
print("\nNumber of duplicate rows:", dup_count)

# Data types of each column
print("\nData types:")
print(df.dtypes)

# ============================================================
# STEP 3: Data Cleaning
# ============================================================
print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Drop duplicate records if any
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"Dropped {dup_count} duplicate rows. New shape: {df.shape}")
else:
    print("No duplicates to drop.")

# Handle missing values
# Strategy: numeric -> median; categorical -> mode (most frequent)
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in [np.float64, np.int64]:
            # Numeric column: fill with median
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing numeric values in '{col}' with median = {median_val}")
        else:
            # Categorical column: fill with mode
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing categorical values in '{col}' with mode = {mode_val}")

# (Optional) Check for out-of-range or invalid values (domain knowledge needed)
# For example, drop rows with non-positive ages or impossible values
if 'age' in df.columns:
    invalid_ages = df[(df['age'] <= 0) | (df['age'] > 120)]
    if not invalid_ages.empty:
        print(f"Dropping {len(invalid_ages)} rows with invalid 'age' values.")
        df = df[(df['age'] > 0) & (df['age'] <= 120)]

print("Shape after cleaning:", df.shape)

# ============================================================
# STEP 4: Identify Target and Features
# ============================================================
print("\n" + "="*60)
print("FEATURE/TARGET SELECTION")
print("="*60)

# Common target column names
possible_targets = [
    "target", "Target", "heart_disease", "HeartDisease", "output", "Outcome"
]

target_column = None
for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    # If not found automatically, list columns and ask user
    print("Target column not found automatically. Columns available:", df.columns.tolist())
    target_column = input("Enter the name of the target column: ")

print(f"Identified target column: '{target_column}'")

# Separate features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

print("\nFeature columns:", X.columns.tolist())
print("Class distribution in target:")
print(y.value_counts())

# ============================================================
# STEP 5: Feature Types
# ============================================================
print("\n" + "="*60)
print("FEATURE TYPES")
print("="*60)

# Identify numeric and categorical features automatically
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ============================================================
# STEP 6: Preprocessing Pipelines
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING PIPELINES")
print("="*60)

# Pipeline for numeric features: standard scaling
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# Pipeline for categorical features: one-hot encoding
categorical_transformer = Pipeline(steps=[
  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

print("Preprocessor configured with StandardScaler for numeric and OneHotEncoder for categorical features.")

# ============================================================
# STEP 7: Train/Test Split
# ============================================================
print("\n" + "="*60)
print("TRAIN/TEST SPLIT")
print("="*60)

# Split data: 80% train, 20% test, stratify by target to preserve class ratios.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples; Test set: {len(X_test)} samples")

# ============================================================
# STEP 8: Model Training & Evaluation
# ============================================================
print("\n" + "="*60)
print("MODEL TRAINING AND EVALUATION")
print("="*60)

# Define models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
trained_pipelines = {}

for name, model in models.items():
    print("\n" + "-"*60)
    print(f"Training {name}...")
    
    # Create a pipeline: preprocessing + model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Store results
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })
    trained_pipelines[name] = pipeline
    
    # Display metrics
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# STEP 9: Summary of Model Performance
# ============================================================
print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)
results_df = pd.DataFrame(results)
# Print a neat table of results
print(results_df.to_string(index=False, formatters={
    "Accuracy": "{:.4f}".format,
    "Precision": "{:.4f}".format,
    "Recall": "{:.4f}".format,
    "F1-Score": "{:.4f}".format
}))

# ============================================================
# STEP 10: Confusion Matrices
# ============================================================
print("\n" + "="*60)
print("CONFUSION MATRICES")
print("="*60)

# Determine best model by accuracy
best_idx = results_df["Accuracy"].idxmax()
best_model_name = results_df.loc[best_idx, "Model"]
best_model = trained_pipelines[best_model_name]
print(f"Best model by accuracy: {best_model_name}")

# Plot confusion matrix for each model
for name, pipeline in trained_pipelines.items():
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()  # Display the plot

# ============================================================
# STEP 11: Save Best Model and Sample Predictions
# ============================================================
print("\n" + "="*60)
print("SAMPLE PREDICTIONS WITH BEST MODEL")
print("="*60)

# Predict on first 10 samples of test set
sample_X = X_test.head(10)
sample_y = y_test.head(10)
predictions = best_model.predict(sample_X)

comparison_df = pd.DataFrame({
    "Actual": list(sample_y.values),
    "Predicted": list(predictions)
})
print(comparison_df)

# (Optional) Save the best model to disk for later use
import joblib
model_filename = "best_heart_model.pkl"
joblib.dump(best_model, model_filename)
print(f"\nBest model saved to '{model_filename}'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
