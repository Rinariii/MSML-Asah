import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

RAW_PATH = r"../loan_dataset_raw/bank_transactions_data.csv"
OUT_DIR = "preprocessing"
OUT_FILE = f"{OUT_DIR}/loan_clean.csv"

def load_data():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di lokasi: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"Dataset Loaded: {df.shape}")
    return df

def process_data(df):
    df = df.copy()

    # 1. Hapus Duplikat
    df.drop_duplicates(inplace=True)

    # 2. Hapus Kolom ID
    df.drop('IP Address', axis=1, inplace=True, errors='ignore')
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    df.drop(columns=id_columns, inplace=True)

    # 3. Impute Missing Values (Median)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # 4. Handle Outliers (IQR -> Median)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        median = df[col].median()
        df[col] = np.where((df[col] < lower) | (df[col] > upper), median, df[col])

    # 5. Binning
    if "CustomerAge" in df.columns:
        bins_age = [0, 18, 40, 60, np.inf]
        labels_age = ["Remaja", "Dewasa Muda", "Dewasa", "Lansia"]
        df["Age_Binned"] = pd.cut(df["CustomerAge"], bins=bins_age, labels=labels_age)

    if "TransactionAmount" in df.columns:
        df["Amount_Binned"] = pd.qcut(
            df["TransactionAmount"], q=4,
            labels=["Kecil", "Sedang", "Besar", "Sangat Besar"],
            duplicates="drop"
        )

    # 6. Encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df

def scale_and_pca(df):
    X = df.copy()

    # Standard Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca

def save_result(X_pca):
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df_pca = df_pca.astype(float)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    df_pca.to_csv(OUT_FILE, index=False)
def main():
    df = load_data()
    df = process_data(df)
    X_pca = scale_and_pca(df)
    save_result(X_pca)

if __name__ == "__main__":
    main()
