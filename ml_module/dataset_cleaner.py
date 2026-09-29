import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


CSV_PATH = (
    r"C:\Users\HP\Desktop\Stego\ml_module\output\probability_analysis"
    r"\probability_intelligence_final_dataset.csv"
)

OUTPUT_PATH = (
    r"C:\Users\HP\Desktop\Stego\ml_module\output\probability_analysis"
    r"\probability_intelligence_final_dataset_cleaned.csv"
)


def clean_dataset(input_path, output_path):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    original_shape = df.shape

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    for column in numeric_columns:
        values = df[column]

        finite_values = values[
            np.isfinite(values)
        ]

        if finite_values.empty:
            df[column] = 0.0
            continue

        positive_infinity_mask = np.isposinf(values)

        if positive_infinity_mask.any():
            df.loc[
                positive_infinity_mask,
                column,
            ] = finite_values.max() + 1.0

        df[column] = df[column].replace(
            -np.inf,
            np.nan,
        )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if numeric_columns:
        imputer = SimpleImputer(
            strategy="median"
        )

        df[numeric_columns] = imputer.fit_transform(
            df[numeric_columns]
        )

    remaining_missing_columns = df.columns[
        df.isna().any()
    ].tolist()

    for column in remaining_missing_columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna("unknown")
        else:
            df[column] = df[column].fillna(0)

    if df.isna().any().any():
        raise ValueError(
            "Missing values remain after cleaning."
        )

    numeric_values = df.select_dtypes(
        include=[np.number]
    ).to_numpy()

    if numeric_values.size > 0:
        if not np.isfinite(numeric_values).all():
            raise ValueError(
                "Non-finite numeric values remain."
            )

    df.to_csv(
        output_path,
        index=False,
    )

    gaussian_columns = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in (
                "gaussian",
                "empirical_cdf",
                "_pdf",
                "_cdf",
            )
        )
    ]

    print("Dataset cleaning completed.")
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(
        f"Gaussian/PDF/CDF columns retained: "
        f"{len(gaussian_columns)}"
    )
    print(f"Original dataset preserved: {input_path}")
    print(f"Cleaned dataset saved: {output_path}")

    return df


if __name__ == "__main__":
    clean_dataset(
        CSV_PATH,
        OUTPUT_PATH,
    )