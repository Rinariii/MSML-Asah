import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

RAW_PATH = r"../loan_dataset_raw/bank_transactions_data.csv"
OUT_DIR = "preprocessing"
OUT_FILE = f"{OUT_DIR}/loan_clean.csv"


def load_data():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di lokasi: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print("Dataset Loaded:", df.shape)
    return df


def drop_duplicates(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    return df


def drop_id_columns(df):
    df = df.copy()

    if "IP Address" in df.columns:
        df.drop("IP Address", axis=1, inplace=True)

    id_columns = [col for col in df.columns if "id" in col.lower()]
    df.drop(columns=id_columns, inplace=True, errors="ignore")
    print("Kolom ID di-drop:", id_columns)

    return df


def binning(df):
    df = df.copy()

    if "CustomerAge" in df.columns:
        bins_age = [0, 18, 40, 60, np.inf]
        labels_age = ["Remaja", "Dewasa Muda", "Dewasa", "Lansia"]

        df["Age_Binned"] = pd.cut(
            df["CustomerAge"],
            bins=bins_age,
            labels=labels_age,
            include_lowest=True
        )

        df["Age_Encoded"] = LabelEncoder().fit_transform(df["Age_Binned"].astype(str))

    if "TransactionAmount" in df.columns:
        df["Amount_Binned"] = pd.qcut(
            df["TransactionAmount"],
            q=4,
            labels=["Kecil", "Sedang", "Besar", "Sangat Besar"],
            duplicates="drop"
        )

        df["Amount_Encoded"] = LabelEncoder().fit_transform(df["Amount_Binned"].astype(str))

    return df


def encode_categorical(df):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df


def feature_scaling(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def handle_missing(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def handle_outliers(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        median = df[col].median()

        df[col] = np.where((df[col] < lower) | (df[col] > upper), median, df[col])

    return df


def save(df):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    df.to_csv(OUT_FILE, index=False)
    print("Saved:", OUT_FILE)


def main():
    df = load_data()
    df = drop_duplicates(df)
    df = drop_id_columns(df)
    df = binning(df)               
    df = encode_categorical(df)    
    df = handle_missing(df)
    df = handle_outliers(df)
    df = feature_scaling(df)
    save(df)
    print("Preprocessing Selesai.")


if __name__ == "__main__":
    main()
