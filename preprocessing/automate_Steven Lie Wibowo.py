import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_FILE = os.path.join(SCRIPT_DIR, "loan_clean.csv")

RAW_PATH = os.path.join(SCRIPT_DIR, "..", "loan_dataset_raw", "bank_transactions_data.csv")

def load_data():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {RAW_PATH}")
    
    df = pd.read_csv(RAW_PATH)
    print(f"Dataset Loaded: {df.shape}")
    return df

def run_preprocessing(df):    
    # 1. Menghapus Data Duplikat
    df.drop_duplicates(inplace=True)

    # 2. Membersihkan Kolom ID
    df.drop('IP Address', axis=1, inplace=True, errors='ignore')
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    df.drop(columns=id_columns, inplace=True)

    # 3. Impute Data Kosong
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # 4. Deteksi dan Penanganan Outlier
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR

        median = df[col].median()
        df[col] = np.where((df[col] < batas_bawah) | (df[col] > batas_atas), median, df[col])

    # 5. Binning
    if "CustomerAge" in df.columns:
        bins_age = [0, 18, 40, 60, np.inf]
        labels_age = ["Remaja", "Dewasa Muda", "Dewasa", "Lansia"]
        df["Age_Binned"] = pd.cut(df["CustomerAge"], bins=bins_age, labels=labels_age)

    if "TransactionAmount" in df.columns:
        df["Amount_Binned"] = pd.qcut(
            df["TransactionAmount"],
            q=4,
            labels=["Kecil", "Sedang", "Besar", "Sangat Besar"],
            duplicates="drop"
        )

    # 6. Encoding Data Kategorikal
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    X = df.copy() 

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return df 

def save_result(df):

    df.to_csv(OUT_FILE, index=False)
    
def main():
    try:
        df = load_data()
        df_processed = run_preprocessing(df)
        save_result(df_processed)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)

if __name__ == "__main__":
    main()
