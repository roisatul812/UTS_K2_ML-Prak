from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "output" / "02_train_scaled.csv"
TEST_PATH = BASE_DIR / "output" / "03_test_scaled.csv"
OUTPUT_DIR = BASE_DIR / "output" / "smote"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===== 2. LOAD DATA HASIL PREPROCESSING =====

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

print("=== DATA HASIL PREPROCESSING ===")
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)
print("X_test  :", X_test.shape)
print("y_test  :", y_test.shape)


# ===== 3. DISTRIBUSI CLASS SEBELUM SMOTE =====

class_before = y_train.value_counts().sort_index()

print("\n=== DISTRIBUSI CLASS SEBELUM SMOTE ===")
print(class_before)


# ===== 4. SMOTE =====

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

class_after = pd.Series(y_train_smote).value_counts().sort_index()

print("\n=== DISTRIBUSI CLASS SETELAH SMOTE ===")
print(class_after)


# Simpan distribusi class untuk dokumentasi laporan
distribution_df = pd.DataFrame({
    "Class": class_before.index,
    "Before_SMOTE": class_before.values,
    "After_SMOTE": [
        class_after.get(cls, 0)
        for cls in class_before.index
    ]
})

distribution_df.to_csv(
    OUTPUT_DIR / "class_distribution_smote.csv",
    index=False
)


# ===== 5. DECISION TREE =====

model = DecisionTreeClassifier(random_state=42)

model.fit(
    X_train_smote,
    y_train_smote
)


# ===== 6. PREDIKSI DATA TESTING =====
# Testing TIDAK di-SMOTE.

y_pred = model.predict(X_test)


# ===== 7. CONFUSION MATRIX =====

cm = confusion_matrix(y_test, y_pred)

print("\n=== CONFUSION MATRIX ===")
print(cm)


# ===== 8. METRIK EVALUASI =====
accuracy = accuracy_score(y_test, y_pred)

precision_macro = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

# Weighted juga disimpan sebagai informasi tambahan.
precision_weighted = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n=== HASIL EVALUASI SMOTE + DECISION TREE ===")
print(f"Accuracy          : {accuracy:.4f}")
print(f"Precision (Macro)  : {precision_macro:.4f}")
print(f"Recall (Macro)     : {recall_macro:.4f}")
print(f"F1-Score (Macro)   : {f1_macro:.4f}")

print("\n=== METRIK WEIGHTED (TAMBAHAN) ===")
print(f"Precision (Weighted): {precision_weighted:.4f}")
print(f"Recall (Weighted)   : {recall_weighted:.4f}")
print(f"F1-Score (Weighted) : {f1_weighted:.4f}")


# ===== 9. CLASSIFICATION REPORT =====

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\n=== CLASSIFICATION REPORT ===")
print(report)


# ===== 10. SIMPAN HASIL METRIK =====

metrics_df = pd.DataFrame({
    "Method": ["SMOTE"],
    "Model": ["Decision Tree"],
    "Accuracy": [accuracy],
    "Precision_Macro": [precision_macro],
    "Recall_Macro": [recall_macro],
    "F1_Macro": [f1_macro],
    "Precision_Weighted": [precision_weighted],
    "Recall_Weighted": [recall_weighted],
    "F1_Weighted": [f1_weighted],
})

metrics_df.to_csv(
    OUTPUT_DIR / "metrics_smote.csv",
    index=False
)


# ===== 11. SIMPAN CLASSIFICATION REPORT =====

with open(
    OUTPUT_DIR / "classification_report_smote.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# ===== 12. SIMPAN HASIL PREDIKSI =====

prediction_df = pd.DataFrame({
    "Actual_Class": y_test.values,
    "Predicted_Class": y_pred
})

prediction_df.to_csv(
    OUTPUT_DIR / "predictions_smote.csv",
    index=False
)


# ===== 13. SIMPAN CONFUSION MATRIX =====

pd.DataFrame(
    cm,
    index=[f"Actual_{cls}" for cls in sorted(y_test.unique())],
    columns=[f"Predicted_{cls}" for cls in sorted(y_test.unique())]
).to_csv(
    OUTPUT_DIR / "confusion_matrix_smote.csv"
)


# ===== 14. VISUALISASI CONFUSION MATRIX =====

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=sorted(y_test.unique())
).plot(ax=ax)

ax.set_title("Confusion Matrix - SMOTE + Decision Tree")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "confusion_matrix_smote.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ===== 15. SELESAI =====

print("\n=== PROSES SMOTE SELESAI ===")
print("Output disimpan di:")
print(OUTPUT_DIR)