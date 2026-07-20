import pandas as pd
import calendar


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
        self.calendar_df = None


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
    def create_lag_features(
        self,
        lag_days: list[int] | None = None,
        group_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Create calendar-aware lag features for historical demand.

        Lag values are matched using the exact historical date
        within each unique demand series.

        Parameters
        ----------
        lag_days : list[int] | None
            Number of calendar days to lag.
            Default: [1, 7, 14, 28]

        group_columns : list[str] | None
            Columns defining a unique demand series.
            Default: ["store", "style", "color", "size"]

        Returns
        -------
        pd.DataFrame
            Dataset containing lag features.
        """

        if lag_days is None:
            lag_days = [1, 7, 14, 28]

        if group_columns is None:
            group_columns = [
                "store",
                "style",
                "color",
                "size"
            ]

        # -----------------------------------
        # Validate required columns
        # -----------------------------------
        required_columns = (
            ["date", "qty"]
            + group_columns
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # -----------------------------------
        # Ensure date is datetime
        # -----------------------------------
        self.df["date"] = pd.to_datetime(
            self.df["date"],
            errors="coerce"
        )

        if self.df["date"].isna().any():
            raise ValueError(
                "Invalid or missing dates found."
            )

        # -----------------------------------
        # Sort data
        # -----------------------------------
        self.df = (
            self.df
            .sort_values(
                group_columns + ["date"]
            )
            .reset_index(drop=True)
        )

        # -----------------------------------
        # Create calendar-aware lag features
        # -----------------------------------
        base_lookup = self.df[
            group_columns + ["date", "qty"]
        ].copy()

        for lag in lag_days:

            lag_lookup = base_lookup.copy()

            # Move historical date forward so it matches
            # the future row that should receive this lag.
            lag_lookup["date"] = (
                lag_lookup["date"]
                + pd.Timedelta(days=lag)
            )

            lag_column = f"lag_{lag}"

            lag_lookup = lag_lookup.rename(
                columns={"qty": lag_column}
            )

            self.df = self.df.merge(
                lag_lookup,
                on=group_columns + ["date"],
                how="left",
                validate="one_to_one"
            )

        self.feature_df = self.df.copy()

        return self.feature_df
    def analyze_demand_sparsity(
        self,
        levels: dict[str, list[str]] | None = None
    ) -> pd.DataFrame:
        """
        Analyze demand sparsity at different forecasting granularities.

        This method compares how frequently demand is observed
        for different combinations such as style, store-style,
        store-style-color, and full SKU level.

        Parameters
        ----------
        levels : dict[str, list[str]] | None
            Forecasting levels to analyze.

        Returns
        -------
        pd.DataFrame
            Summary of demand sparsity at each forecasting level.
        """

        # -----------------------------------
        # Default forecasting levels
        # -----------------------------------
        if levels is None:

            levels = {
                "style": [
                    "style"
                ],

                "store_style": [
                    "store",
                    "style"
                ],

                "store_style_color": [
                    "store",
                    "style",
                    "color"
                ],

                "store_style_color_size": [
                    "store",
                    "style",
                    "color",
                    "size"
                ]
            }

        # -----------------------------------
        # Validate required columns
        # -----------------------------------
        if "date" not in self.df.columns:

            raise ValueError(
                "Required column 'date' not found."
            )

        if "qty" not in self.df.columns:

            raise ValueError(
                "Required column 'qty' not found."
            )

        # Ensure datetime
        self.df["date"] = pd.to_datetime(
            self.df["date"],
            errors="coerce"
        )

        if self.df["date"].isna().any():

            raise ValueError(
                "Invalid dates found in dataset."
            )

        results = []

        # -----------------------------------
        # Analyze each forecasting level
        # -----------------------------------
        for level_name, group_columns in levels.items():

            missing_columns = [
                column
                for column in group_columns
                if column not in self.df.columns
            ]

            if missing_columns:

                print(
                    f"Skipping {level_name}. "
                    f"Missing columns: {missing_columns}"
                )

                continue

            # -----------------------------------
            # Aggregate demand at this level
            # -----------------------------------
            level_df = (
                self.df
                .groupby(
                    group_columns + ["date"],
                    as_index=False
                )
                .agg(
                    qty=("qty", "sum")
                )
            )

            # -----------------------------------
            # Statistics for each demand series
            # -----------------------------------
            series_stats = (
                level_df
                .groupby(group_columns)
                .agg(
                    first_observed_date=(
                        "date",
                        "min"
                    ),
                    last_observed_date=(
                        "date",
                        "max"
                    ),
                    observed_days=(
                        "date",
                        "nunique"
                    ),
                    total_qty=(
                        "qty",
                        "sum"
                    )
                )
                .reset_index()
            )

            # -----------------------------------
            # Calculate observation window
            # -----------------------------------
            series_stats["calendar_days"] = (
                (
                    series_stats["last_observed_date"]
                    - series_stats["first_observed_date"]
                ).dt.days
                + 1
            )

            # -----------------------------------
            # Demand density
            # -----------------------------------
            series_stats["demand_density"] = (
                series_stats["observed_days"]
                / series_stats["calendar_days"]
            )

            # -----------------------------------
            # Average gap between observations
            # -----------------------------------
            series_stats["avg_gap_days"] = (
                series_stats["calendar_days"]
                / series_stats["observed_days"]
            )

            # -----------------------------------
            # Create summary
            # -----------------------------------
            results.append(
                {
                    "forecast_level": level_name,

                    "number_of_series":
                        len(series_stats),

                    "avg_observed_days":
                        round(
                            series_stats[
                                "observed_days"
                            ].mean(),
                            2
                        ),

                    "median_observed_days":
                        round(
                            series_stats[
                                "observed_days"
                            ].median(),
                            2
                        ),

                    "avg_calendar_days":
                        round(
                            series_stats[
                                "calendar_days"
                            ].mean(),
                            2
                        ),

                    "avg_demand_density_pct":
                        round(
                            series_stats[
                                "demand_density"
                            ].mean()
                            * 100,
                            2
                        ),

                    "median_demand_density_pct":
                        round(
                            series_stats[
                                "demand_density"
                            ].median()
                            * 100,
                            2
                        ),

                    "avg_gap_days":
                        round(
                            series_stats[
                                "avg_gap_days"
                            ].mean(),
                            2
                        )
                }
            )

        return pd.DataFrame(results)
    def create_complete_calendar(self) -> pd.DataFrame:
        """
        Create a continuous daily calendar for every Store × Style series.

        Missing sales dates are inserted and Qty is filled with 0.
        Static attributes (category, description, channel) are
        forward/backward filled within each series.

        Returns
        -------
        pd.DataFrame
            Continuous daily sales dataset.
        """

        # -------------------------------------------------------
        # Validate required columns
        # -------------------------------------------------------
        required_columns = [
            "date",
            "store",
            "style",
            "qty"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # -------------------------------------------------------
        # Copy data
        # -------------------------------------------------------
        df = self.df.copy()

        df["date"] = pd.to_datetime(df["date"])

        completed_series = []

        grouped = df.groupby(
            ["store", "style"],
            sort=False
        )

        print(
            f"\nCreating calendars for {len(grouped)} Store × Style series..."
        )

        # -------------------------------------------------------
        # Create calendar for each Store × Style
        # -------------------------------------------------------
        for (store, style), group in grouped:

            group = (
                group
                .sort_values("date")
                .reset_index(drop=True)
            )

            first_date = group["date"].min()
            last_date = group["date"].max()

            # Create continuous date range
            calendar = pd.DataFrame({
                "date": pd.date_range(
                    start=first_date,
                    end=last_date,
                    freq="D"
                )
            })

            calendar["store"] = store
            calendar["style"] = style

            # Merge with sales
            calendar = calendar.merge(
                group,
                on=["date", "store", "style"],
                how="left"
            )
            # Remove unreliable state column
            if "state" in calendar.columns:
                calendar = calendar.drop(columns=["state"])

            # Fill Qty with zero
            calendar["qty"] = (
                calendar["qty"]
                .fillna(0)
                .astype(int)
            )

            # Fill static attributes
            static_columns = [
                "category",
                "description",
                "channel"
            ]

            for column in static_columns:

                if column in calendar.columns:

                    calendar[column] = (
                        calendar[column]
                        .ffill()
                        .bfill()
                    )

            # Do NOT fill state because it is inconsistent
            # Keep it as-is for now.

            completed_series.append(calendar)

        # -------------------------------------------------------
        # Combine all Store × Style series
        # -------------------------------------------------------
        self.calendar_df = (
            pd.concat(
                completed_series,
                ignore_index=True
            )
            .sort_values(
                ["store", "style", "date"]
            )
            .reset_index(drop=True)
        )

        print("\nCalendar Completion Finished")
        print("--------------------------------")
        print(f"Original Rows : {len(df):,}")
        print(f"Calendar Rows : {len(self.calendar_df):,}")
        print(f"Added Rows    : {len(self.calendar_df)-len(df):,}")

        return self.calendar_df
    def validate_calendar(self):    

        print("\nValidating Calendar...")
        print("-" * 40)

        duplicates = self.calendar_df.duplicated(
            subset=["date", "store", "style"]
        ).sum()

        print(f"Duplicate rows : {duplicates}")

        missing_qty = self.calendar_df["qty"].isna().sum()

        print(f"Missing Qty : {missing_qty}")

        print(f"Total Rows : {len(self.calendar_df):,}")

        print("Validation Complete")
    def create_lag_features(
        self,
        lags=[1, 7, 14, 28]
    ):
        """
        Create lag features for each Store × Style series.

        Parameters
        ----------
        lags : list
            List of lag periods.

        Returns
        -------
        pd.DataFrame
            Dataset with lag features.
        """

        # -------------------------------------------------------
        # Validate
        # -------------------------------------------------------

        if not hasattr(self, "calendar_df"):
            raise ValueError(
                "Run create_complete_calendar() first."
            )

        # -------------------------------------------------------
        # Copy calendar
        # -------------------------------------------------------

        df = self.calendar_df.copy()

        # -------------------------------------------------------
        # Sort data
        # -------------------------------------------------------

        df = (
            df
            .sort_values(
                ["store", "style", "date"]
            )
            .reset_index(drop=True)
        )

        # -------------------------------------------------------
        # Group Store × Style
        # -------------------------------------------------------

        grouped = df.groupby(
            ["store", "style"]
        )

        # -------------------------------------------------------
        # Create Lag Features
        # -------------------------------------------------------

        print("\nCreating Lag Features...")

        for lag in lags:

            print(f"Creating lag_{lag}")

            df[f"lag_{lag}"] = (
                grouped["qty"]
                .shift(lag)
            )

        # -------------------------------------------------------
        # Save
        # -------------------------------------------------------

        self.feature_df = df

        print("\nLag Feature Creation Completed")
        print("-----------------------------------")
        print(f"Shape : {self.feature_df.shape}")

        return self.feature_df
    def validate_lag_features(self):

        if not hasattr(self, "feature_df"):
            raise ValueError(
                "Run create_lag_features() first."
            )

        print("\nValidating Lag Features...")
        print("--------------------------------")

        lag_columns = [
            col
            for col in self.feature_df.columns
            if col.startswith("lag_")
        ]

        print("\nMissing Values")

        for column in lag_columns:

            print(
                f"{column} : "
                f"{self.feature_df[column].isna().sum():,}"
            )

        duplicates = self.feature_df.duplicated(
            subset=["date", "store", "style"]
        ).sum()

        print(f"\nDuplicate Rows : {duplicates}")

        print("\nValidation Completed")