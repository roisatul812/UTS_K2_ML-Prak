import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## ===== 1. LOAD DATASET =====
dataset_name = "Multiclass Diabetes Dataset"
dataset_path = "../database/Multiclass-Diabetes-Dataset.csv"

df = pd.read_csv(dataset_path)

print("=== DATASET INFO ===")
print("Nama dataset  :", dataset_name)
print("Jumlah baris  :", df.shape[0])
print("Jumlah kolom  :", df.shape[1])

print("\n=== 5 DATA PERTAMA ===")
print(df.head())

print("\n=== INFORMASI DATA ===")
print(df.info())

print("\n=== STATISTIK DESKRIPTIF ===")
print(df.describe())

print("\n=== NAMA KOLOM ===")
print(df.columns.tolist())

# ===== 2. CEK MISSING VALUE =====
print("\n=== MISSING VALUE ===")
print(df.isnull().sum())

# ===== 3. CEK DUPLICATE =====
print("\n=== DUPLICATE ===")
print("Jumlah duplicate :", df.duplicated().sum())

# ===== 4. DISTRIBUSI CLASS =====
print("\n=== DISTRIBUSI CLASS ===")
print(df["Class"].value_counts().sort_index())

# ===== 5. PEMISAHAN FITUR DAN TARGET =====
X = df.drop(columns=["Class"])
y = df["Class"]

print("\n=== FITUR ===")
print(X.columns.tolist())

print("\n=== TARGET ===")
print(y.name)

# ===== 6. SPLIT DATA =====
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== DATA SPLIT ===")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

print("\n=== CLASS TRAINING ===")
print(y_train.value_counts().sort_index())

print("\n=== CLASS TESTING ===")
print(y_test.value_counts().sort_index())

# ===== 7. DETEKSI OUTLIER =====
numeric_features = X_train.columns.tolist()

outlier_results = []

for column in numeric_features:
    Q1 = X_train[column].quantile(0.25)
    Q3 = X_train[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_count = (
        (X_train[column] < lower_bound) |
        (X_train[column] > upper_bound)
    ).sum()

    outlier_results.append({
        "Feature": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": outlier_count
    })

outlier_df = pd.DataFrame(outlier_results)

print("\n=== OUTLIER ===")
print(
    outlier_df[
        ["Feature", "Lower Bound", "Upper Bound", "Outlier Count"]
    ]
)

# ===== 8. HANDLING OUTLIER =====
for column in numeric_features:
    Q1 = X_train[column].quantile(0.25)
    Q3 = X_train[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    X_train[column] = X_train[column].clip(
        lower=lower_bound,
        upper=upper_bound
    )

    X_test[column] = X_test[column].clip(
        lower=lower_bound,
        upper=upper_bound
    )

print("\n=== HASIL HANDLING OUTLIER ===")
for column in numeric_features:
    Q1 = X_train[column].quantile(0.25)
    Q3 = X_train[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    count = (
        (X_train[column] < lower_bound) |
        (X_train[column] > upper_bound)
    ).sum()

    print(column, ":", count)

# ===== 9. STANDARDISASI =====
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)

print("\n=== HASIL STANDARDISASI ===")
print(X_train_scaled.head())

# ===== 10. SIMPAN DATA TRAINING DAN TESTING =====
train_data = X_train_scaled.copy()
train_data["Class"] = y_train.values

test_data = X_test_scaled.copy()
test_data["Class"] = y_test.values

train_data.to_csv(
    "../output/02_train_scaled.csv",
    index=False
)

test_data.to_csv(
    "../output/03_test_scaled.csv",
    index=False
)

print("\n=== DATA TERSIMPAN ===")
print("Training : ../output/02_train_scaled.csv")
print("Testing  : ../output/03_test_scaled.csv")

# ===== 11. SIMPAN HASIL OUTLIER =====
outlier_df.to_csv(
    "../output/01_outlier_results.csv",
    index=False
)

print("Outlier  : ../output/01_outlier_results.csv")

# ===== 12. SELESAI =====
print("\n=== PREPROCESSING SELESAI ===")