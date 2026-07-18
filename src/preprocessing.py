from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class RetailPreprocessor:
    """
    Handles data loading, validation, cleaning,
    aggregation, and preprocessing.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        mapping_path: str = "config/column_mapping.yaml",
        validation_path: str = "config/validation_rules.yaml",
    ) -> None:

        self.config = self.load_yaml(config_path)
        self.mapping = self.load_yaml(mapping_path)
        self.validation = self.load_yaml(validation_path)

        self.df = None
        self.column_map = {}

    def load_yaml(self, filepath: str) -> dict[str, Any]:
        """
        Load a YAML configuration file.

        Parameters
        ----------
        filepath : str
            Path to the YAML configuration file.

        Returns
        -------
        dict[str, Any]
            Parsed YAML configuration.
        """

        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            if config is None:
                raise ValueError(
                    f"Configuration file is empty: {path}"
                )

            return config

        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML file: {path}"
            ) from e
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load a dataset from a CSV or Excel file.

        Parameters
        ----------
        filepath : str
            Path to the dataset file.

        Returns
        -------
        pd.DataFrame
            Loaded dataset as a pandas DataFrame.
        """

        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        suffix = path.suffix.lower()

        try:
            if suffix == ".csv":
                df = pd.read_csv(path)

            elif suffix in [".xls", ".xlsx"]:
                df = pd.read_excel(path)

            else:
                raise ValueError(
                    f"Unsupported file format: {suffix}. "
                    "Supported formats are .csv, .xls, and .xlsx."
                )

        except Exception as e:
            raise ValueError(
                f"Failed to load dataset: {e}"
            ) from e

        if df.empty:
            raise ValueError("Dataset is empty.")

        self.df = df

        return df
    def detect_columns(self) -> dict[str, str]:
        """
        Detect dataset columns using aliases defined in
        column_mapping.yaml.

        Returns
        -------
        dict[str, str]
            Mapping between standard column names and
            dataset column names.

        Raises
        ------
        ValueError
            If a required column cannot be found.
        """
        if self.df is None:
            raise ValueError("Dataframe is not loaded. Call load_data() first.")
        column_map = {}
        #loop through every standard field 
        for standard_name, config in self.mapping.items():
            aliases = config.get("aliases", [])
            matched = False 
            #search every aliases 
            for alias in aliases:
                if alias in self.df.columns:
                    column_map[standard_name] = alias
                    matched = True
                    break
            #required column missing
            if not aliases:
                raise ValueError(f"No aliases defined for '{standard_name}' in column_mapping.yaml")
        self.column_map = column_map
        return column_map
    def generate_data_quality_report(self) -> dict[str, Any]:
        """
        Generate a data quality report based on the loaded dataset.

        Returns
        -------
        dict[str, Any]
            Data quality report containing information about
            missing values, duplicates, and data types.

        """
        date_column = self.column_map.get("date")
        if self.df is None:
            raise ValueError("Dataframe is not loaded. Call load_data() first.")

        report = {
            "dataset_info": {"rows": len(self.df), "columns": len(self.df.columns),"memory_usage":round(self.df.memory_usage(deep=True).sum()/1024**2, 2)},
            "required_columns": {"date": True, "store_id": True, "product_id": True, "quantity_sold": True, "sale_amount": True},
            "optional_columns": {col: col in self.df.columns for col in self.mapping.keys() if not self.mapping[col].get("required", False)},
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicates": self.df.duplicated().sum(),
            "date_range": (self.df[date_column].min(), self.df[date_column].max()) if date_column else None,
            "unique_counts": {col: self.df[col].nunique() for col in self.df.columns},
            "ignored_columns": [col for col in self.df.columns if col not in self.column_map.values()],
            "data_types": self.df.dtypes.apply(lambda x: x.name).to_dict(),
        }

        return report
    def rename_columns(self) -> pd.DataFrame:
        """
        Rename dataset columns to standard names based on
        the detected column mapping.

        Returns
        -------
        pd.DataFrame
            DataFrame with renamed columns.
        """
        if self.df is None:
            raise ValueError("Dataframe is not loaded. Call load_data() first.")
        if not self.column_map:
            raise ValueError("Column mapping is empty. Call detect_columns() first.")
         # Reverse the mapping:
        # {"date": "Inv Date"} → {"Inv Date": "date"}
        rename_mappings={original: standard for standard, original in self.column_map.items()}
        #Create a renamed copy 
        self.processed_df = self.df.rename(columns=rename_mappings)
        return self.processed_df
    def remove_empty_columns(self) -> pd.DataFrame:
        """
        Remove empty columns such as 'Unnamed: 13'
        created during Excel export.

        Returns
        -------
        pd.DataFrame
            Dataset without unnamed columns.
        """

        if self.processed_df is None:
            raise ValueError(
                "Processed dataset not found. "
                "Run rename_columns() first."
            )

        unnamed_columns = [
            col
            for col in self.processed_df.columns
            if col.startswith("Unnamed")
        ]

        self.processed_df = self.processed_df.drop(
            columns=unnamed_columns
        )

        return self.processed_df
    def standardize_text_columns(self) -> pd.DataFrame:
        """
        Standardize all text columns by:

        - Removing leading/trailing spaces
        - Replacing multiple spaces with a single space
        - Converting text to uppercase

        Returns
        -------
        pd.DataFrame
            Standardized dataset.
        """

        if self.processed_df is None:
            raise ValueError(
                "Processed dataset not found. "
                "Run rename_columns() first."
            )

        # Identify text columns
        text_columns = self.processed_df.select_dtypes(include=["object"]).columns

        for column in text_columns:

            self.processed_df[column] = (

                self.processed_df[column]

                .astype(str)

                .str.strip()

                .str.replace(r"\s+", " ", regex=True)

                .str.upper()

            )

        return self.processed_df
    def convert_data_types(self) -> pd.DataFrame:
        """
        Convert dataset columns to the required
        data types defined in validation_rules.yaml.

        Returns
        -------
        pd.DataFrame
            Dataset with standardized data types.
        """

        if self.processed_df is None:
            raise ValueError(
                "Processed dataset not found. "
                "Run rename_columns() first."
            )

        for column, rules in self.validation.items():

            # Skip special sections like required_columns
            if not isinstance(rules, dict):
                continue

            if column not in self.processed_df.columns:
                continue

            datatype = rules.get("datatype")

            if datatype == "datetime":

                self.processed_df[column] = pd.to_datetime(
                    self.processed_df[column],
                    errors="coerce"
                )

            elif datatype == "integer":

                self.processed_df[column] = pd.to_numeric(
                    self.processed_df[column],
                    errors="coerce"
                ).astype("Int64")

            elif datatype == "float":

                self.processed_df[column] = pd.to_numeric(
                    self.processed_df[column],
                    errors="coerce"
                )

        return self.processed_df
    def business_validation(self) -> dict[str, Any]:
        """
        Validate the processed dataset against business rules
        defined in validation_rules.yaml.

        Returns
        -------
        dict[str, Any]
            Validation report.
        """

        if self.processed_df is None:
            raise ValueError(
                "Processed dataset not found. "
                "Run rename_columns() first."
            )

        validation_summary = {
            "status": "PASSED",
            "total_violations": 0,
            "columns": {}
        }

        for column, rules in self.validation.items():

            # Skip non-column sections
            if not isinstance(rules, dict):
                continue

            # Skip columns not present
            if column not in self.processed_df.columns:
                continue

            violations = set()

            # -------------------------------
            # Required column validation
            # -------------------------------
            if rules.get("required", False):

                missing_rows = self.processed_df[
                    self.processed_df[column].isna()
                ].index.tolist()

                violations.update(missing_rows)

            # -------------------------------
            # Minimum value validation
            # -------------------------------
            min_value = rules.get("min")

            if min_value is not None:

                invalid_rows = self.processed_df[
                    self.processed_df[column] < min_value
                ].index.tolist()

                violations.update(invalid_rows)

            # -------------------------------
            # Maximum value validation
            # -------------------------------
            max_value = rules.get("max")

            if max_value is not None:

                invalid_rows = self.processed_df[
                    self.processed_df[column] > max_value
                ].index.tolist()

                violations.update(invalid_rows)

            # -------------------------------
            # Allowed values validation
            # -------------------------------
            allowed_values = rules.get("allowed_values")

            if allowed_values is not None:

                invalid_rows = self.processed_df[
                    ~self.processed_df[column].isin(allowed_values)
                ].index.tolist()

                violations.update(invalid_rows)

            # -------------------------------
            # Future date validation
            # -------------------------------
            if column == "date":

                future_rows = self.processed_df[
                    self.processed_df[column] > pd.Timestamp.today()
                ].index.tolist()

                violations.update(future_rows)

            validation_summary["columns"][column] = {
                "count": len(violations),
                "violations": sorted(list(violations))
            }

            validation_summary["total_violations"] += len(violations)

        if validation_summary["total_violations"] > 0:
            validation_summary["status"] = "FAILED"

        return validation_summary
    def aggregate_daily_sales(self) -> pd.DataFrame:
        """
        Aggregate transactional ERP sales into daily demand.

        The ERP stores one row per barcode transaction.
        This method converts transactional data into
        daily sales suitable for forecasting.

        Returns
        -------
        pd.DataFrame
            Daily aggregated sales dataset.
        """

        if self.processed_df is None:
            raise ValueError(
                "Processed dataset not found. "
                "Run preprocessing before aggregation."
            )

        # -----------------------------
        # Read grouping columns
        # -----------------------------
        group_columns = self.config.get(
            "aggregation_level",
            []
        )

        if not group_columns:
            raise ValueError(
                "aggregation_level not found in config.yaml"
            )

        # -----------------------------
        # Validate grouping columns
        # -----------------------------
        missing_columns = [
            column
            for column in group_columns
            if column not in self.processed_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing grouping columns: {missing_columns}"
            )

        # -----------------------------
        # Read aggregation rules
        # -----------------------------
        aggregation_rules = self.config.get(
            "aggregation_rules",
            {}
        )

        if not aggregation_rules:
            raise ValueError(
                "aggregation_rules not found in config.yaml"
            )

        # -----------------------------
        # Keep only existing columns
        # -----------------------------
        aggregation_rules = {
            column: rule
            for column, rule in aggregation_rules.items()
            if column in self.processed_df.columns
        }

        # -----------------------------
        # Aggregate daily sales
        # -----------------------------
        self.aggregated_df = (
            self.processed_df
            .groupby(
                group_columns,
                as_index=False
            )
            .agg(aggregation_rules)
        )

        # -----------------------------
        # Sort dataset
        # -----------------------------
        self.aggregated_df = (
            self.aggregated_df
            .sort_values(group_columns)
            .reset_index(drop=True)
        )

        return self.aggregated_df
    def prepare_modeling_data(
        self,
        group_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Prepare modeling dataset at the Store x Style x Date level.

        The detailed aggregated sales data may contain multiple
        color and size combinations for the same store and style.
        This method combines them into daily Store x Style demand.

        Parameters
        ----------
        group_columns : list[str] | None
            Columns defining the modeling granularity.

        Returns
        -------
        pd.DataFrame
            Store x Style daily demand dataset.
        """

        if self.aggregated_df is None:
            raise ValueError(
                "Aggregated dataset not found. "
                "Run aggregate_daily_sales() first."
            )

        if group_columns is None:
            group_columns = [
                "date",
                "store",
                "style"
            ]

        # Validate required columns
        missing_columns = [
            column
            for column in group_columns
            if column not in self.aggregated_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing modeling columns: {missing_columns}"
            )

        # Define aggregation rules
        aggregation_rules = {
            "qty": "sum"
        }

        # Keep useful business dimensions
        optional_dimensions = {
            "category": "first",
            "description": "first",
            "state": "first",
            "channel": "first"
        }

        for column, rule in optional_dimensions.items():

            if column in self.aggregated_df.columns:
                aggregation_rules[column] = rule

        # Create Store x Style daily dataset
        modeling_df = (
            self.aggregated_df
            .groupby(
                group_columns,
                as_index=False
            )
            .agg(aggregation_rules)
        )

        # Sort chronologically
        modeling_df = (
            modeling_df
            .sort_values(
                ["store", "style", "date"]
            )
            .reset_index(drop=True)
        )

        self.modeling_df = modeling_df

        return self.modeling_df