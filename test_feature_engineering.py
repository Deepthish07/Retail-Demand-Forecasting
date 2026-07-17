import pandas as pd

from src.feature_engineering import FeatureEngineer


# ==========================================
# STEP 1: Load aggregated dataset
# ==========================================

df = pd.read_excel(
    "data/processed/aggregated_sales.xlsx"
)

print("Original Shape:")
print(df.shape)


# ==========================================
# STEP 2: Initialize FeatureEngineer
# ==========================================

feature_engineer = FeatureEngineer(df)


# ==========================================
# STEP 3: Create calendar features
# ==========================================

feature_df = feature_engineer.create_calendar_features()

print("\nAfter Calendar Features:")
print(feature_df.shape)


# ==========================================
# STEP 4: Create lag features
# IMPORTANT: Reassign feature_df here
# ==========================================

feature_df = feature_engineer.create_lag_features()

print("\nAfter Lag Features:")
print(feature_df.shape)


# ==========================================
# STEP 5: Check all columns
# ==========================================

print("\nAll Columns:")

print(
    feature_df.columns.tolist()
)


# ==========================================
# STEP 6: Define lag columns
# ==========================================

lag_columns = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28"
]


# ==========================================
# STEP 7: Verify lag columns exist
# ==========================================

missing_columns = [
    column
    for column in lag_columns
    if column not in feature_df.columns
]

if missing_columns:

    raise ValueError(
        f"Lag columns were not created: "
        f"{missing_columns}"
    )


# ==========================================
# STEP 8: Display lag feature sample
# ==========================================

print("\nLag Feature Sample:")

print(
    feature_df[
        [
            "date",
            "store",
            "style",
            "color",
            "size",
            "qty",
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28"
        ]
    ].head(20)
)


# ==========================================
# STEP 9: Missing lag values
# ==========================================

print("\nMissing Lag Values:")

print(
    feature_df[
        lag_columns
    ].isna().sum()
)


# ==========================================
# STEP 10: Missing lag percentage
# ==========================================

print("\nMissing Lag Percentage:")

missing_percentage = (
    feature_df[
        lag_columns
    ]
    .isna()
    .mean()
    .mul(100)
    .round(2)
    )
sparsity_report = (
    feature_engineer
    .analyze_demand_sparsity()
)


print("\nDemand Sparsity Analysis:")

print(
    sparsity_report.to_string(
        index=False
    )
)


print(missing_percentage)
sparsity_report = (
    feature_engineer
    .analyze_demand_sparsity()
)


print("\nDemand Sparsity Analysis:")

print(
    sparsity_report.to_string(
        index=False
    )
)