import pandas as pd
from pathlib import Path


class CSVDataLoader:
    """
    Intelligent CSV reader with automatic type inference and normalization.

    This class provides reproducible data loading by automatically detecting and
    converting column types appropriately for statistical analysis:
    - Parses date columns automatically
    - Preserves integer columns using pandas nullable Int64
    - Converts object columns to categorical for memory efficiency
    - Provides detailed logging of type detection and conversions

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file to load.
    log : bool, default True
        Whether to print log messages during loading process.

    Attributes
    ----------
    filepath : Path
        Resolved path to the CSV file.
    log : bool
        Whether logging is enabled.
    log_messages : list of str
        Accumulated log messages from the loading process.
    df : pd.DataFrame or None
        Loaded and processed DataFrame (None until load() is called).

    Examples
    --------
    >>> loader = CSVDataLoader('data/surgical_outcomes.csv')
    >>> df, log_messages = loader.load()
    === Loading CSV file: data/surgical_outcomes.csv ===
    Initial shape: (165, 45)
    Parsed date/time columns: ['Surgery Date']
    Converted to Int64 (nullable integer): ['Age', 'Length of Stay']
    Converted 12 object columns to categorical.

    Notes
    -----
    The automatic type inference follows these rules:
    1. Columns with >80% datetime-parseable values become datetime64
    2. Float columns where all non-null values are whole numbers become Int64
    3. All remaining object columns become categorical

    This approach balances type accuracy with memory efficiency and is
    optimized for clinical datasets with mixed data types.
    """

    def __init__(self, filepath: str | Path, log: bool = True):
        self.filepath = Path(filepath)
        self.log = log
        self.log_messages = []
        self.df = None

    def _log(self, message: str):
        """
        Record a log message and optionally print it.

        Parameters
        ----------
        message : str
            Message to log.
        """
        if self.log:
            print(message)
        self.log_messages.append(message)

    def load(self, **read_csv_kwargs):
        """
        Load CSV file with automatic dtype inference and normalization.

        This method performs the complete loading pipeline:
        1. Read CSV with pandas default inference
        2. Detect and parse datetime columns
        3. Convert integer-like floats to nullable Int64
        4. Convert object columns to categorical

        Parameters
        ----------
        **read_csv_kwargs : dict, optional
            Additional keyword arguments passed to pd.read_csv().
            Common options include sep, encoding, na_values.

        Returns
        -------
        df : pd.DataFrame
            Loaded and type-normalized DataFrame.
        log_messages : list of str
            List of all log messages generated during loading.

        Examples
        --------
        >>> loader = CSVDataLoader('data.csv')
        >>> df, logs = loader.load()
        >>> print(df.dtypes)

        >>> # Load with custom parameters
        >>> df, logs = loader.load(encoding='latin1', na_values=['NA', 'Missing'])

        See Also
        --------
        pandas.read_csv : Underlying CSV reading function.
        """
        self._log(f"=== Loading CSV file: {self.filepath} ===")

        # Initial read – allow pandas to infer dtypes, but low_memory=False helps consistency
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

    def _parse_dates(self):
        """
        Automatically detect and parse datetime columns.

        Samples up to 50 values from each object/string column and attempts
        datetime parsing. If >80% of sampled values successfully parse,
        the entire column is converted to datetime64.

        Returns
        -------
        self : CSVDataLoader
            Returns self for method chaining.

        Notes
        -----
        Only object and string dtype columns are considered for datetime parsing.
        The 80% threshold balances detecting true date columns while avoiding
        false positives from columns with date-like strings.
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

    def _enforce_integer_types(self):
        """
        Convert float columns representing integers to nullable Int64 dtype.

        Checks all float columns to determine if they contain only integer values
        (allowing for NaN). If so, converts to pandas nullable Int64, which
        preserves integer type while supporting missing data.

        Returns
        -------
        self : CSVDataLoader
            Returns self for method chaining.

        Notes
        -----
        This addresses a common pandas issue where integer columns with missing
        values are read as float64. The nullable Int64 dtype (capital I) was
        introduced in pandas 0.24 to properly represent integers with NA values.

        Integer preservation is important for count variables (length of stay,
        number of complications) where the distinction between 0 and missing
        has clinical significance.
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

    def _convert_objects_to_categoricals(self):
        """
        Convert all object columns to categorical dtype for memory efficiency.

        Categorical dtype reduces memory usage and can improve performance for
        columns with repeated values (common in clinical data with categorical
        variables like race, procedure type, comorbidities).

        Returns
        -------
        self : CSVDataLoader
            Returns self for method chaining.

        Notes
        -----
        All remaining object columns are converted to categorical. This should
        be done after datetime parsing to avoid converting date strings to
        categorical incorrectly.

        For typical clinical datasets, this conversion can reduce memory usage
        by 50-90% for categorical columns.
        """
        before_shape = self.df.shape
        object_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        for col in object_cols:
            self.df[col] = self.df[col].astype("category")
        self._log(f"Converted {len(object_cols)} object columns to categorical.")
        self._log(f"Categorical conversion complete | Shape unchanged: {before_shape}")
        return self
