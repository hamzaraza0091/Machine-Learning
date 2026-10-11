# ============================================================
# SMS SPAM CLASSIFICATION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("SMS SPAM CLASSIFICATION")
print("=" * 60)

df = pd.read_csv("sms.csv", encoding="latin-1")

print("\nFirst 5 rows:")
print(df.head())

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nOriginal Columns:")
print(df.columns)

# ============================================================
# 2. CLEAN DATASET
# ============================================================

# Keep only the useful columns
df = df[["v1", "v2"]]

# Rename columns
df.columns = ["label", "message"]

print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nCleaned Columns:")
print(df.columns)

# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

print("\nClass Distribution:")
print(df["label"].value_counts())

print("\nClass Distribution Percentage:")
print(df["label"].value_counts(normalize=True) * 100)

# ============================================================
# 5. DATASET INFORMATION
# ============================================================

print("\nDataset Information:")
print(df.info())

# ============================================================
# 6. CONVERT LABELS
# ============================================================

# ham = 0
# spam = 1

df["label_encoded"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nEncoded Labels:")
print(df[["label", "label_encoded"]].head())

# ============================================================
# 7. FEATURES AND TARGET
# ============================================================

X = df["message"]
y = df["label_encoded"]

# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ============================================================
# 9. TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 60)
print("TF-IDF VECTORIZATION")
print("=" * 60)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

# Fit only on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining TF-IDF Shape:")
print(X_train_tfidf.shape)

print("\nTesting TF-IDF Shape:")
print(X_test_tfidf.shape)

# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING NAIVE BAYES MODEL")
print("=" * 60)

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")

# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_tfidf)

# ============================================================
# 12. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)

# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot()

plt.title("SMS Spam Classification - Confusion Matrix")
plt.tight_layout()
plt.show()

# ============================================================
# 15. TEST CUSTOM SMS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOM SMS TESTING")
print("=" * 60)


def predict_sms(message):
    """
    Predict whether an SMS is Ham or Spam.
    """

    # Convert message into TF-IDF
    message_tfidf = vectorizer.transform([message])

    # Make prediction
    prediction = model.predict(message_tfidf)[0]

    # Get probability
    probabilities = model.predict_proba(message_tfidf)[0]

    spam_probability = probabilities[1] * 100
    ham_probability = probabilities[0] * 100

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    print("\nSMS:")
    print(message)

    print("\nPrediction:", result)

    print(f"Ham Probability:  {ham_probability:.2f}%")
    print(f"Spam Probability: {spam_probability:.2f}%")

    print("-" * 60)


# ============================================================
# 16. TEST EXAMPLES
# ============================================================

predict_sms(
    "Congratulations! You have won a free iPhone. "
    "Click here to claim your prize."
)

predict_sms(
    "Hey, are you coming to class today?"
)

predict_sms(
    "URGENT! You have won 1000 dollars. "
    "Call now to claim your prize."
)

predict_sms(
    "Can you send me the project file when you get home?"
)

# ============================================================
# 17. INTERACTIVE SMS TESTING
# ============================================================

print("\n" + "=" * 60)
print("INTERACTIVE SMS SPAM DETECTOR")
print("=" * 60)

print("\nType an SMS to classify it.")
print("Type 'exit' to stop.")

while True:

    user_message = input("\nEnter SMS: ")

    if user_message.lower() == "exit":
        print("\nProgram finished.")
        break

    predict_sms(user_message)