import pandas as pd


class FeatureEngineer:
    """
    Handles feature engineering for the
    retail demand forecasting pipeline.

    This class creates features used by the
    machine learning model, including:

    - Calendar features
    - Lag features
    - Rolling statistics
    - Retail demand features
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the FeatureEngineer.

        Parameters
        ----------
        df : pd.DataFrame
            Aggregated daily sales dataset.
        """

        if df is None or df.empty:
            raise ValueError(
                "Input DataFrame cannot be empty."
            )

        # Create a copy to avoid modifying
        # the original aggregated dataset
        self.df = df.copy()

        # This will hold the final feature dataset
        self.feature_df = None


    def create_calendar_features(self) -> pd.DataFrame:
        """
        Create calendar-based features from the date column.

        Returns
        -------
        pd.DataFrame
            Dataset containing additional calendar features.
        """

        if "date" not in self.df.columns:
            raise ValueError(
                "Required column 'date' not found in dataset."
            )

        # Ensure date is datetime
        self.df["date"] = pd.to_datetime(
            self.df["date"],
            errors="coerce"
        )

        # Check for invalid dates
        if self.df["date"].isna().any():
            raise ValueError(
                "Invalid or missing values found in the date column."
            )

        # Calendar features
        self.df["year"] = self.df["date"].dt.year

        self.df["month"] = self.df["date"].dt.month

        self.df["quarter"] = self.df["date"].dt.quarter

        self.df["week_of_year"] = (
            self.df["date"]
            .dt
            .isocalendar()
            .week
            .astype("int")
        )

        self.df["day_of_month"] = (
            self.df["date"].dt.day
        )

        self.df["day_of_week"] = (
            self.df["date"].dt.dayofweek
        )

        self.df["is_weekend"] = (
            self.df["day_of_week"]
            .isin([5, 6])
            .astype(int)
        )

        self.df["is_month_start"] = (
            self.df["date"]
            .dt
            .is_month_start
            .astype(int)
        )

        self.df["is_month_end"] = (
            self.df["date"]
            .dt
            .is_month_end
            .astype(int)
        )

        self.feature_df = self.df.copy()

        return self.feature_df