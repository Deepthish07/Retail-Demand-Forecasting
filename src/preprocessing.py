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