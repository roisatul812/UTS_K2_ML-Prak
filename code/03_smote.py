from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ===== 1. PATH FILE =====
# Path dihitung dari lokasi file ini, sehingga script bisa dijalankan
# dari folder mana pun.
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "output"
TRAIN_PATH = INPUT_DIR / "02_train_scaled.csv"
TEST_PATH = INPUT_DIR / "03_test_scaled.csv"
OUTPUT_DIR = INPUT_DIR / "smote"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Pemetaan label sesuai deskripsi dataset (Mendeley Data): jumlah data
# 96 non-diabetic, 40 predict-diabetic, 128 diabetic.
CLASS_NAMES = {0: "Non-Diabetic", 1: "Predict-Diabetic", 2: "Diabetic"}


# ===== 2. LOAD DATA HASIL PREPROCESSING =====
# Split TIDAK dibuat ulang agar sama dengan Person 1 dan Person 2.

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

print("=== DATA HASIL PREPROCESSING ===")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)


# ===== 3. DISTRIBUSI CLASS SEBELUM SMOTE =====

before_smote = y_train.value_counts().sort_index()

print("\n=== DISTRIBUSI CLASS TRAINING SEBELUM SMOTE ===")
print(before_smote)


# ===== 4. SMOTE (HANYA PADA DATA TRAINING) =====

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

after_smote = y_train_smote.value_counts().sort_index()

print("\n=== DISTRIBUSI CLASS TRAINING SESUDAH SMOTE ===")
print(after_smote)
print("\nX_train_smote :", X_train_smote.shape)
print("y_train_smote :", y_train_smote.shape)

print("\n=== DISTRIBUSI CLASS TESTING (TIDAK DI-RESAMPLING) ===")
print(y_test.value_counts().sort_index())

# Simpan perbandingan distribusi kelas
distribution_df = pd.DataFrame({
    "Class": before_smote.index,
    "Nama Class": [CLASS_NAMES[c] for c in before_smote.index],
    "Sebelum SMOTE": before_smote.values,
    "Sesudah SMOTE": after_smote.reindex(before_smote.index).values,
    "Testing (tidak di-resampling)": y_test.value_counts().sort_index().reindex(before_smote.index).values,
})
distribution_df.to_csv(
    OUTPUT_DIR / "01_smote_class_distribution.csv", index=False
)

# Grafik distribusi kelas sebelum dan sesudah SMOTE
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
labels = [f"{c}\n{CLASS_NAMES[c]}" for c in before_smote.index]
for ax, counts, title in zip(
    axes,
    [before_smote, after_smote],
    ["Sebelum SMOTE (Training)", "Sesudah SMOTE (Training)"],
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
plt.savefig(OUTPUT_DIR / "02_smote_class_distribution.png", dpi=150)
plt.close()


# ===== 5. TRAINING DECISION TREE =====

dt_smote = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_smote.fit(X_train_smote, y_train_smote)

print("\n=== DECISION TREE (SMOTE) ===")
print("Parameter    :", dt_smote.get_params())
print("Kedalaman    :", dt_smote.get_depth())
print("Jumlah daun  :", dt_smote.get_n_leaves())


# ===== 6. PREDIKSI PADA DATA TESTING ASLI =====

y_pred = dt_smote.predict(X_test)

pred_df = X_test.copy()
pred_df["Class Aktual"] = y_test.values
pred_df["Class Prediksi"] = y_pred
pred_df.to_csv(OUTPUT_DIR / "03_smote_predictions.csv", index=False)


# ===== 7. EVALUASI =====

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
cm_df.to_csv(OUTPUT_DIR / "04_smote_confusion_matrix.csv")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[f"{c}\n{CLASS_NAMES[c]}" for c in labels_order],
)
fig, ax = plt.subplots(figsize=(7, 5.5))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title("Confusion Matrix - SMOTE + Decision Tree", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_smote_confusion_matrix.png", dpi=150)
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

print("\n=== METRIK EVALUASI (SMOTE + DECISION TREE) ===")
for name, value in metrics.items():
    print(f"{name:22s}: {value:.4f}")

metrics_df = pd.DataFrame(
    {"Metrik": list(metrics.keys()), "Nilai": list(metrics.values())}
)
metrics_df.to_csv(OUTPUT_DIR / "06_smote_metrics.csv", index=False)

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
per_class_df.to_csv(OUTPUT_DIR / "07_smote_metrics_per_class.csv", index=False)

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

with open(OUTPUT_DIR / "08_smote_classification_report.txt", "w", encoding="utf-8") as f:
    f.write("SMOTE + Decision Tree\n")
    f.write("=" * 60 + "\n")
    f.write(report)


# ===== 8. FEATURE IMPORTANCE =====

importance_df = (
    pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": dt_smote.feature_importances_,
    })
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
print("\n=== FEATURE IMPORTANCE ===")
print(importance_df.round(4).to_string(index=False))
importance_df.to_csv(OUTPUT_DIR / "09_smote_feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color="#4C78A8")
ax.set_xlabel("Importance")
ax.set_title("Feature Importance - SMOTE + Decision Tree")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_smote_feature_importance.png", dpi=150)
plt.close()


# ===== 9. SELESAI =====

print("\n=== SMOTE + DECISION TREE SELESAI ===")
print("Output tersimpan di :", OUTPUT_DIR)