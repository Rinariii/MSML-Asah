import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = SCRIPT_DIR
OUT_FILE_CSV = os.path.join(OUT_DIR, "loan_clean.csv")

RAW_PATH = os.path.join(SCRIPT_DIR, "..", "loan_dataset_raw", "bank_transactions_data.csv")

# ==============================================================================

def load_data():
    print(f"Script Location: {SCRIPT_DIR}")
    print(f"Looking for data at: {os.path.abspath(RAW_PATH)}")
    
    if not os.path.exists(RAW_PATH):
        parent_dir = os.path.dirname(SCRIPT_DIR)
        print(f"Isi folder parent ({parent_dir}): {os.listdir(parent_dir)}")
        raise FileNotFoundError(f"FATAL: Dataset tidak ditemukan di {RAW_PATH}")
            
    df = pd.read_csv(RAW_PATH)
    print(f"Dataset Loaded: {df.shape}")
    return df

def process_data(df):
    df = df.copy()
    
    # 1. Hapus Duplikat & ID
    df.drop_duplicates(inplace=True)
    df.drop('IP Address', axis=1, inplace=True, errors='ignore')
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    df.drop(columns=id_columns, inplace=True)
    
    # 2. Impute Missing Values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # 3. Outliers
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        median = df[col].median()
        df[col] = np.where((df[col] < lower) | (df[col] > upper), median, df[col])

    # 4. Binning
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

    # 5. Encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df

def scale_and_pca(df):
    # Hanya ambil numerik
    X = df.select_dtypes(include=[np.number]).copy()

    # Standard Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def save_result(X_pca):
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

    # Simpan CSV
    df_pca.to_csv(OUT_FILE_CSV, index=False)
    print(f"File CSV BERHASIL disimpan di: {OUT_FILE_CSV}")

def main():
    try:
        df = load_data()
        df = process_data(df)
        X_pca = scale_and_pca(df)
        save_result(X_pca)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)

if __name__ == "__main__":
    main()
