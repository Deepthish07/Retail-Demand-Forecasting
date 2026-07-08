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