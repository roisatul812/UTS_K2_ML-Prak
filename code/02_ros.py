import os

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Path dihitung dari lokasi file ini, sehingga script bisa dijalankan
# dari folder mana pun (python code/02_random_oversampling.py atau cd code).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "output")
OUTPUT_DIR = os.path.join(INPUT_DIR, "random_oversampling")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CLASS_NAMES = {0: "Non-Diabetic", 1: "Diabetic", 2: "Predict-Diabetic"}

## ===== 1. LOAD DATA HASIL PREPROCESSING (PERSON 1) =====
# Data training dan testing sudah di-split 80:20 (stratify) dan di-standardisasi
# oleh 01_preprocessing.py. Split TIDAK dibuat ulang agar sama dengan Person 3.
train_data = pd.read_csv(os.path.join(INPUT_DIR, "02_train_scaled.csv"))
test_data = pd.read_csv(os.path.join(INPUT_DIR, "03_test_scaled.csv"))

X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]
X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

print("=== DATA HASIL PREPROCESSING ===")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

## ===== 2. DISTRIBUSI CLASS SEBELUM ROS =====
before_ros = y_train.value_counts().sort_index()

print("\n=== DISTRIBUSI CLASS TRAINING SEBELUM ROS ===")
print(before_ros)

## ===== 3. RANDOM OVERSAMPLING (HANYA PADA DATA TRAINING) =====
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

after_ros = y_train_ros.value_counts().sort_index()

print("\n=== DISTRIBUSI CLASS TRAINING SESUDAH ROS ===")
print(after_ros)
print("\nX_train_ros :", X_train_ros.shape)
print("y_train_ros :", y_train_ros.shape)

print("\n=== DISTRIBUSI CLASS TESTING (TIDAK DI-RESAMPLING) ===")
print(y_test.value_counts().sort_index())

# Simpan perbandingan distribusi kelas
distribution_df = pd.DataFrame({
    "Class": before_ros.index,
    "Nama Class": [CLASS_NAMES[c] for c in before_ros.index],
    "Sebelum ROS": before_ros.values,
    "Sesudah ROS": after_ros.reindex(before_ros.index).values,
    "Testing (tidak di-resampling)": y_test.value_counts().sort_index().reindex(before_ros.index).values,
})
distribution_df.to_csv(
    os.path.join(OUTPUT_DIR, "01_ros_class_distribution.csv"), index=False
)

# Grafik distribusi kelas sebelum dan sesudah ROS
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
labels = [f"{c}\n{CLASS_NAMES[c]}" for c in before_ros.index]
for ax, counts, title in zip(
    axes,
    [before_ros, after_ros],
    ["Sebelum ROS (Training)", "Sesudah ROS (Training)"],
):
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title(title)
    ax.set_xlabel("Class")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            int(bar.get_height()),
            ha="center",
            va="bottom",
        )
axes[0].set_ylabel("Jumlah Data")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_ros_class_distribution.png"), dpi=150)
plt.close()

## ===== 4. TRAINING DECISION TREE =====
dt_ros = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_ros.fit(X_train_ros, y_train_ros)

print("\n=== DECISION TREE (ROS) ===")
print("Parameter    :", dt_ros.get_params())
print("Kedalaman    :", dt_ros.get_depth())
print("Jumlah daun  :", dt_ros.get_n_leaves())

## ===== 5. PREDIKSI PADA DATA TESTING ASLI =====
y_pred = dt_ros.predict(X_test)

pred_df = X_test.copy()
pred_df["Class Aktual"] = y_test.values
pred_df["Class Prediksi"] = y_pred
pred_df.to_csv(os.path.join(OUTPUT_DIR, "03_ros_predictions.csv"), index=False)

## ===== 6. EVALUASI =====
labels_order = sorted(y_train.unique())

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
cm_df = pd.DataFrame(
    cm,
    index=[f"Aktual {c}" for c in labels_order],
    columns=[f"Prediksi {c}" for c in labels_order],
)

print("\n=== CONFUSION MATRIX ===")
print(cm_df)
cm_df.to_csv(os.path.join(OUTPUT_DIR, "04_ros_confusion_matrix.csv"))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[f"{c}\n{CLASS_NAMES[c]}" for c in labels_order],
)
fig, ax = plt.subplots(figsize=(7, 5.5))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title("Confusion Matrix - ROS + Decision Tree", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_ros_confusion_matrix.png"), dpi=150)
plt.close()

# Metrik keseluruhan
accuracy = accuracy_score(y_test, y_pred)

metrics = {
    "Accuracy": accuracy,
    "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
    "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
    "F1-Score (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
    "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "F1-Score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
}

print("\n=== METRIK EVALUASI (ROS + DECISION TREE) ===")
for name, value in metrics.items():
    print(f"{name:22s}: {value:.4f}")

metrics_df = pd.DataFrame(
    {"Metrik": list(metrics.keys()), "Nilai": list(metrics.values())}
)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "06_ros_metrics.csv"), index=False)

# Metrik per kelas
per_class_df = pd.DataFrame({
    "Class": labels_order,
    "Nama Class": [CLASS_NAMES[c] for c in labels_order],
    "Precision": precision_score(y_test, y_pred, average=None, labels=labels_order, zero_division=0),
    "Recall": recall_score(y_test, y_pred, average=None, labels=labels_order, zero_division=0),
    "F1-Score": f1_score(y_test, y_pred, average=None, labels=labels_order, zero_division=0),
    "Support": [int((y_test == c).sum()) for c in labels_order],
})
print("\n=== METRIK PER CLASS ===")
print(per_class_df.round(4).to_string(index=False))
per_class_df.to_csv(os.path.join(OUTPUT_DIR, "07_ros_metrics_per_class.csv"), index=False)

# Classification report
report = classification_report(
    y_test,
    y_pred,
    labels=labels_order,
    target_names=[f"{c} - {CLASS_NAMES[c]}" for c in labels_order],
    digits=4,
    zero_division=0,
)
print("\n=== CLASSIFICATION REPORT ===")
print(report)

with open(os.path.join(OUTPUT_DIR, "08_ros_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write("Random Oversampling + Decision Tree\n")
    f.write("=" * 60 + "\n")
    f.write(report)

## ===== 7. FEATURE IMPORTANCE =====
importance_df = (
    pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": dt_ros.feature_importances_,
    })
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
print("\n=== FEATURE IMPORTANCE ===")
print(importance_df.round(4).to_string(index=False))
importance_df.to_csv(os.path.join(OUTPUT_DIR, "09_ros_feature_importance.csv"), index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color="#4C78A8")
ax.set_xlabel("Importance")
ax.set_title("Feature Importance - ROS + Decision Tree")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_ros_feature_importance.png"), dpi=150)
plt.close()

## ===== 8. SELESAI =====
print("\n=== RANDOM OVERSAMPLING + DECISION TREE SELESAI ===")
print("Output tersimpan di :", os.path.normpath(OUTPUT_DIR))