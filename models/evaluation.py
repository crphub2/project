import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pickle

# Load data
df = pd.read_csv("data/Crop_recommendation.csv")

# Features and target
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Naive Bayes": GaussianNB()
}

accuracies = {}
best_model = None
best_score = 0

# Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_score:
        best_model = model
        best_score = acc
        best_name = name
        best_preds = preds

# Save best model
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save classification report
with open("models/classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_name}\n\n")
    f.write(classification_report(y_test, best_preds))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"{best_name} Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("models/confusion_matrix_rf.png")
plt.close()

# Accuracy comparison plot
plt.figure(figsize=(8, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)
plt.ylim(0.8, 1.0)
for i, acc in enumerate(accuracies.values()):
    plt.text(i, acc + 0.002, f"{acc:.3f}", ha='center')
plt.tight_layout()
plt.savefig("models/comparison_chart.png")
plt.close()
