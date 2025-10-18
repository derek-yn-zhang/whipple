import pandas as pd
from pathlib import Path


class CSVDataLoader:
    """
    Intelligent CSV reader that:
      - Parses date columns automatically
      - Preserves integer columns (using pandas nullable Int64)
      - Converts all object columns to categorical
      - Provides reproducible logging of detected and cast types
    """

    def __init__(self, filepath: str | Path, log: bool = True):
        self.filepath = Path(filepath)
        self.log = log
        self.log_messages = []
        self.df = None

    def _log(self, message: str):
        if self.log:
            print(message)
        self.log_messages.append(message)

    # -------------------------------------------------------------------------
    # 1. LOADING & PARSING
    # -------------------------------------------------------------------------
    def load(self, **read_csv_kwargs):
        """
        Reads CSV with automatic dtype inference and type normalization.
        """
        self._log(f"=== Loading CSV file: {self.filepath} ===")

        # Initial read — allow pandas to infer dtypes, but low_memory=False helps consistency
        self.df = pd.read_csv(self.filepath, low_memory=False, **read_csv_kwargs)
        self._log(f"Initial shape: {self.df.shape}")

        # Detect and parse date columns
        self._parse_dates()

        # Convert numeric columns safely
        self._enforce_integer_types()

        # Convert object columns to categorical
        self._convert_objects_to_categoricals()

        self._log("=== CSV Loading Complete ===")
        return self.df, self.log_messages

    # -------------------------------------------------------------------------
    # 2. DATE PARSING
    # -------------------------------------------------------------------------
    def _parse_dates(self):
        """
        Automatically detect and parse datetime columns.
        Only object or string columns are considered.
        Any column where a majority of non-null values can be parsed as datetimes
        will be converted to datetime dtype.
        """
        before_shape = self.df.shape
        parsed_cols = []

        for col in self.df.columns:
            # Only object/string columns
            if not pd.api.types.is_object_dtype(
                self.df[col]
            ) and not pd.api.types.is_string_dtype(self.df[col]):
                continue

            series = self.df[col].dropna()
            if series.empty:
                continue

            # Sample up to 50 values
            sample = series.sample(min(50, len(series)), random_state=42)
            parsed_sample = pd.to_datetime(sample, errors="coerce")
            parse_success_rate = parsed_sample.notna().mean()

            if parse_success_rate > 0.8:
                # Convert entire column
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                parsed_cols.append(col)

        if parsed_cols:
            self._log(f"Parsed date/time columns: {parsed_cols}")
        else:
            self._log("No date-like columns detected.")

        self._log(f"Date parsing completed | Shape unchanged: {before_shape}")
        return self

    # -------------------------------------------------------------------------
    # 3. INTEGER ENFORCEMENT
    # -------------------------------------------------------------------------
    def _enforce_integer_types(self):
        """
        Convert float columns that represent integers (even with NaN) to pandas nullable Int64.
        """
        before_shape = self.df.shape
        converted = []

        for col in self.df.select_dtypes(include=["float"]).columns:
            series = self.df[col]
            # Check if all non-null values are integer-like
            non_null = series.dropna()
            if (non_null % 1 == 0).all():
                try:
                    self.df[col] = series.astype("Int64")
                    converted.append(col)
                except Exception:
                    pass

        if converted:
            self._log(f"Converted to Int64 (nullable integer): {converted}")
        else:
            self._log("No integer-like float columns detected.")
        self._log(f"Integer enforcement completed | Shape unchanged: {before_shape}")
        return self

    # -------------------------------------------------------------------------
    # 4. OBJECT → CATEGORICAL
    # -------------------------------------------------------------------------
    def _convert_objects_to_categoricals(self):
        """
        Convert all remaining object columns to categorical dtype.
        """
        before_shape = self.df.shape
        object_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        for col in object_cols:
            self.df[col] = self.df[col].astype("category")
        self._log(f"Converted {len(object_cols)} object columns to categorical.")
        self._log(f"Categorical conversion complete | Shape unchanged: {before_shape}")
        return self
