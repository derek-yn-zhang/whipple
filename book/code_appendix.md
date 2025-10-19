# Code

This appendix contains the source code for custom classes used throughout the analysis.

## Data Loading

<details>
<summary>CSVDataLoader</summary>

```python
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
```

</details>

## Data Cleaning

<details>
<summary>DataCleaner</summary>

```python
import pandas as pd


class DataCleaner:
    """
    A modular, user-guided data cleaning class designed for reproducible preprocessing
    of surgical outcomes data.

    Core functionality:
      - Keep only relevant fields (user-defined)
      - Drop highly imbalanced predictors (automated)
      - Derive composite or standardized variables (user-defined functions)
      - Clean categorical predictors and apply ordered categorical types
    """

    def __init__(
        self, df: pd.DataFrame, column_metadata: dict[str, str], log: bool = True
    ):
        """
        Initialize DatasetCleaner with required column metadata.

        Parameters:
            df: DataFrame to clean
            column_metadata: Dictionary mapping column names to types ('response' or 'explanatory')
            log: Whether to print log messages
        """
        self.df = df.copy()
        self.column_metadata = column_metadata
        self.log = log
        self.log_messages = []

    def _log(self, message: str):
        """Print and record a log message if logging is enabled."""
        if self.log:
            print(message)
        self.log_messages.append(message)

    # -------------------------------------------------------------------------
    # 1. DROPPING METHODS
    # -------------------------------------------------------------------------
    def drop_columns(self, columns: list[str]):
        """
        Drop specified columns immediately without any checks or conditions.
        Useful for removing known unusable or irrelevant columns upfront.

        Parameters:
            columns: List of column names to drop
        """
        existing = [c for c in columns if c in self.df.columns]
        if not existing:
            self._log("No specified columns found to drop.")
            return self

        self._log(f"Dropping {len(existing)} specified columns: {existing}")
        before_shape = self.df.shape
        self.df.drop(columns=existing, inplace=True)
        self._log(f"Shape changed from {before_shape} to {self.df.shape}")
        return self

    def keep_datetime_columns(self, columns: list[str] = None):
        """
        Keep only the specified datetime columns, dropping all other datetime columns.
        If no columns specified, drops all datetime columns.
        Auto-detects datetime columns by dtype.

        Parameters:
            columns: List of datetime column names to keep. If None, drops all datetime columns.
        """
        # Identify all datetime columns in the DataFrame
        datetime_cols = self.df.select_dtypes(
            include=["datetime64", "datetime64[ns]", "datetimetz"]
        ).columns.tolist()

        if not datetime_cols:
            self._log("No datetime columns detected in DataFrame.")
            return self

        # If no columns specified, drop all datetime columns
        if columns is None:
            self._log(
                f"No datetime columns specified - dropping all {len(datetime_cols)} datetime columns: {datetime_cols}"
            )
            before_shape = self.df.shape
            self.df.drop(columns=datetime_cols, inplace=True)
            self._log(f"Shape changed from {before_shape} to {self.df.shape}")
            return self

        # Find which specified columns actually exist and are datetime
        existing = [c for c in columns if c in datetime_cols]

        # Find datetime columns to drop (not in keep list)
        to_drop = [c for c in datetime_cols if c not in columns]

        if not to_drop:
            self._log("No datetime columns to drop - all are in the keep list.")
            return self

        self._log(f"Detected {len(datetime_cols)} datetime columns: {datetime_cols}")
        self._log(f"Keeping datetime columns: {existing}")
        self._log(f"Dropping {len(to_drop)} datetime columns: {to_drop}")
        before_shape = self.df.shape
        self.df.drop(columns=to_drop, inplace=True)
        self._log(f"Shape changed from {before_shape} to {self.df.shape}")
        return self

    def keep_response_columns(self, columns: list[str]):
        """
        Keep only specified response/outcome columns, dropping all other response columns.
        Uses column_metadata to identify response columns.
        """
        # Identify all response columns from metadata
        response_cols = [
            col
            for col, col_type in self.column_metadata.items()
            if col_type == "response" and col in self.df.columns
        ]

        if not response_cols:
            self._log("No response columns detected in DataFrame metadata.")
            return self

        # Find which specified columns actually exist and are responses
        existing = [c for c in columns if c in response_cols]

        # Find response columns to drop (not in keep list)
        to_drop = [c for c in response_cols if c not in columns]

        if not to_drop:
            self._log("No response columns to drop - all are in the keep list.")
            return self

        self._log(f"Detected {len(response_cols)} response columns from metadata")
        self._log(f"Keeping response columns: {existing}")
        self._log(f"Dropping {len(to_drop)} response columns: {to_drop}")
        before_shape = self.df.shape
        self.df.drop(columns=to_drop, inplace=True)
        self._log(f"Shape changed from {before_shape} to {self.df.shape}")
        return self

    def keep_explanatory_columns(self, columns: list[str]):
        """
        Keep only specified explanatory/predictor columns, dropping all other explanatory columns.
        Uses column_metadata to identify explanatory columns.
        """
        # Identify all explanatory columns from metadata
        explanatory_cols = [
            col
            for col, col_type in self.column_metadata.items()
            if col_type == "explanatory" and col in self.df.columns
        ]

        if not explanatory_cols:
            self._log("No explanatory columns detected in DataFrame metadata.")
            return self

        # Find which specified columns actually exist and are explanatory
        existing = [c for c in columns if c in explanatory_cols]

        # Find explanatory columns to drop (not in keep list)
        to_drop = [c for c in explanatory_cols if c not in columns]

        if not to_drop:
            self._log("No explanatory columns to drop - all are in the keep list.")
            return self

        self._log(f"Detected {len(explanatory_cols)} explanatory columns from metadata")
        self._log(f"Keeping explanatory columns: {existing}")
        self._log(f"Dropping {len(to_drop)} explanatory columns: {to_drop}")
        before_shape = self.df.shape
        self.df.drop(columns=to_drop, inplace=True)
        self._log(f"Shape changed from {before_shape} to {self.df.shape}")
        return self

    def _is_quasi_categorical(
        self, 
        series: pd.Series, 
        uniqueness_threshold: float = 0.10,
        max_absolute_unique: int = 10
    ) -> bool:
        """
        Multi-criteria approach to determine if an integer column behaves categorically.
        
        Treats as categorical if EITHER:
        1. Has ≤10 unique values (regardless of sample size), OR
        2. Unique/total ratio < 10%
        
        This catches:
        - Binary/ordinal scales (0-5 ratings) → always categorical
        - Low-cardinality counts in large samples (5 unique in 10,000 obs) → categorical
        - Skips high-cardinality even in small samples (100 unique in 200 obs) → continuous
        
        Parameters:
            series: Pandas Series to evaluate
            uniqueness_threshold: Maximum uniqueness ratio (default 0.10)
            max_absolute_unique: Maximum unique values to always treat as categorical (default 10)
        
        Returns:
            True if should be treated as categorical, False if continuous
        """
        n_unique = series.nunique()
        n_total = len(series.dropna())
        
        if n_total == 0:
            return False
        
        # Absolute criterion: very few unique values
        if n_unique <= max_absolute_unique:
            return True
        
        # Relative criterion: low uniqueness ratio
        uniqueness_ratio = n_unique / n_total
        return uniqueness_ratio < uniqueness_threshold

    def drop_imbalanced_variables(
        self, 
        min_minority_count: int = 10,
        uniqueness_threshold: float = 0.10,
        max_absolute_unique: int = 10
    ):
        """
        Drop categorical and quasi-categorical integer variables where the second largest class
        has too few observations for adequate statistical power.
        
        Uses second largest class (not smallest) to allow variables with long tails of rare events,
        which can be merged or handled during analysis. This is important for clinical data where
        rare complications may exist but the main contrast (e.g., None vs Any) is still meaningful.
        
        Integer columns are only checked if they behave like categorical variables based on:
        - Having ≤10 unique values (regardless of sample size), OR
        - Having <10% unique-to-total ratio
        
        Truly continuous count variables are automatically exempt from balance checking.
        
        Parameters:
            min_minority_count: Minimum number of observations required in the second largest class.
                            Default is 10, which is a common rule of thumb for regression modeling
                            (10 events per variable for logistic regression).
            uniqueness_threshold: Maximum uniqueness ratio for integers to be checked (default 0.10).
            max_absolute_unique: Maximum unique values to always treat as categorical (default 10).
        """
        before_shape = self.df.shape
        to_drop = []

        # Categorical columns always checked
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Integer columns - filter to quasi-categorical only
        integer_cols = self.df.select_dtypes(
            include=["int64", "int32", "int16", "int8", "Int64"]
        ).columns.tolist()
        
        quasi_categorical_ints = [
            col for col in integer_cols 
            if self._is_quasi_categorical(
                self.df[col], 
                uniqueness_threshold=uniqueness_threshold,
                max_absolute_unique=max_absolute_unique
            )
        ]
        
        # Track which integer columns are being skipped
        continuous_ints = [col for col in integer_cols if col not in quasi_categorical_ints]
        
        cols_to_check = categorical_cols + quasi_categorical_ints
        
        self._log(
            f"Checking class imbalance for {len(cols_to_check)} categorical and quasi-categorical integer columns"
        )
        if continuous_ints:
            self._log(
                f"Skipping {len(continuous_ints)} continuous integer columns (high cardinality): {continuous_ints}"
            )

        for col in cols_to_check:
            value_counts = self.df[col].value_counts(dropna=True)  # Exclude NaN from counts

            if len(value_counts) == 0:  # All values are NaN
                continue

            if len(value_counts) == 1:  # Only one class - definitely drop
                to_drop.append(col)
                col_type = self.column_metadata.get(col, "unknown")
                counts_str = ", ".join(
                    [f"{val}: {count}" for val, count in value_counts.items()]
                )
                nan_count = self.df[col].isna().sum()
                if nan_count > 0:
                    counts_str += f", NaN: {nan_count}"
                self._log(f"  '{col}' ({col_type}): only one class present")
                self._log(f"    Class counts: {counts_str}")
                continue

            # Check second largest class
            sorted_counts = value_counts.sort_values(ascending=False)
            second_largest_count = sorted_counts.iloc[1]

            col_type = self.column_metadata.get(col, "unknown")

            if second_largest_count < min_minority_count:
                to_drop.append(col)
                # Format value counts for logging
                counts_str = ", ".join(
                    [f"{val}: {count}" for val, count in value_counts.items()]
                )

                # Add NaN count if present
                nan_count = self.df[col].isna().sum()
                if nan_count > 0:
                    counts_str += f", NaN: {nan_count}"

                self._log(
                    f"  '{col}' ({col_type}): second largest class has {second_largest_count} observations (< {min_minority_count})"
                )
                self._log(f"    Class counts: {counts_str}")

        if to_drop:
            self._log(
                f"Dropping {len(to_drop)} imbalanced variables with insufficient second class size"
            )
            self.df.drop(columns=to_drop, inplace=True)
            self._log(f"Shape changed from {before_shape} to {self.df.shape}")
        else:
            self._log("No imbalanced variables detected.")
        return self

    def drop_high_missingness_variables(self, max_missing_rate: float = 0.40):
        """
        Drop variables (both explanatory and response) with excessive missing data
        that would compromise statistical power.

        Parameters:
            max_missing_rate: Maximum proportion of missing values allowed (0 to 1).
                             Default is 0.40 (40%), a common threshold where imputation becomes unreliable
                             and complete case analysis would lose too much data.
        """
        before_shape = self.df.shape
        to_drop = []

        # Check all columns (both explanatory and response)
        all_cols = self.df.columns.tolist()

        self._log(f"Checking missingness for {len(all_cols)} columns")

        for col in all_cols:
            missing_rate = self.df[col].isna().sum() / len(self.df)

            col_type = self.column_metadata.get(col, "unknown")

            if missing_rate > max_missing_rate:
                to_drop.append(col)
                self._log(
                    f"  '{col}' ({col_type}): {missing_rate*100:.1f}% missing (> {max_missing_rate*100:.0f}%)"
                )

        if to_drop:
            self._log(f"Dropping {len(to_drop)} variables with excessive missingness")
            self.df.drop(columns=to_drop, inplace=True)
            self._log(f"Shape changed from {before_shape} to {self.df.shape}")
        else:
            self._log("No variables with excessive missingness detected.")
        return self

    # -------------------------------------------------------------------------
    # 2. DERIVED VARIABLES
    # -------------------------------------------------------------------------
    def derive_variables(
        self,
        derivation_map: dict[str, callable],
        drop_components: dict[str, list[str]] = None,
    ):
        """
        Create new variables based on user-supplied functions and optionally drop component columns.
        Automatically infers the metadata type (response/explanatory) for derived columns based on
        component columns.

        Each function should take the full DataFrame as input and return a Series.

        Parameters:
            derivation_map: Dictionary mapping new column names to derivation functions.
            drop_components: Dictionary mapping new column names to lists of component columns to drop.

        Example:
            derivation_map = {
                "Height (cm)": lambda df: df.apply(standardize_height_to_cm, axis=1),
                "Any Postop Sepsis": lambda df: df.apply(composite_sepsis, axis=1)
            }
            drop_components = {
                "Height (cm)": ["Height", "Height Unit"],
                "Any Postop Sepsis": ["# of Postop Sepsis", "# of Postop Septic Shock"]
            }
        """
        drop_components = drop_components or {}

        for new_col, func in derivation_map.items():
            self._log(f"Deriving variable: '{new_col}'")
            before_shape = self.df.shape
            self.df[new_col] = func(self.df)
            self._log(f"  Added '{new_col}' | Shape: {before_shape} -> {self.df.shape}")

            # Infer metadata type from component columns
            if new_col in drop_components:
                component_types = [
                    self.column_metadata.get(comp)
                    for comp in drop_components[new_col]
                    if comp in self.column_metadata
                ]

                if component_types:
                    # If all components are the same type, use that type
                    unique_types = set(component_types)
                    if len(unique_types) == 1:
                        inferred_type = component_types[0]
                        self.column_metadata[new_col] = inferred_type
                        self._log(
                            f"  Inferred metadata type '{inferred_type}' for '{new_col}' based on component columns"
                        )
                    else:
                        # Mixed types - log warning and default to explanatory
                        self.column_metadata[new_col] = "explanatory"
                        self._log(
                            f"  Warning: Component columns have mixed types {unique_types}. Defaulting '{new_col}' to 'explanatory'"
                        )
                else:
                    # No component metadata found - default to explanatory
                    self.column_metadata[new_col] = "explanatory"
                    self._log(
                        f"  No component metadata found. Defaulting '{new_col}' to 'explanatory'"
                    )
            else:
                # No components specified - default to explanatory
                self.column_metadata[new_col] = "explanatory"
                self._log(
                    f"  No component columns specified. Defaulting '{new_col}' to 'explanatory'"
                )

            # Drop component columns if specified
            if new_col in drop_components:
                components_to_drop = [
                    c for c in drop_components[new_col] if c in self.df.columns
                ]
                if components_to_drop:
                    self._log(f"  Dropping component columns: {components_to_drop}")
                    self.df.drop(columns=components_to_drop, inplace=True)
                    self._log(f"  Shape after dropping components: {self.df.shape}")
                else:
                    self._log(f"  No component columns found to drop for '{new_col}'")

        return self

    # -------------------------------------------------------------------------
    # 3. CLEANING CATEGORICAL PREDICTORS
    # -------------------------------------------------------------------------
    def clean_categorical_predictors(
        self, consolidation_maps: dict[str, dict], ordered_maps: dict[str, list] = None
    ):
        """
        Apply user-supplied mappings to consolidate and order categorical predictors.
        Example:
            consolidation_maps = {
                "Race/Ethnicity": {
                    "Black or African American": "Black",
                    "White": "White",
                    "Some Other Race": "Other"
                }
            }
            ordered_maps = {
                "ASA Classification": [
                    "ASA II - Mild systemic disease",
                    "ASA III - Severe systemic disease",
                    "ASA IV - Severe systemic disease threat to life"
                ]
            }
        """
        before_shape = self.df.shape
        # Apply consolidation
        for column, mapping in consolidation_maps.items():
            if column not in self.df.columns:
                self._log(f"Skipping {column} (not found in DataFrame).")
                continue
            self._log(
                f"Consolidating categories in '{column}' using mapping with {len(mapping)} entries."
            )
            self.df[column] = self.df[column].map(mapping).astype("category")

        # Apply ordering
        if ordered_maps:
            for column, categories in ordered_maps.items():
                if column in self.df.columns:
                    cat_type = pd.api.types.CategoricalDtype(
                        categories=categories, ordered=True
                    )
                    self.df[column] = self.df[column].astype(cat_type)
                    self._log(f"Applied ordered categorical type to '{column}'.")

        self._log(
            f"Finished categorical cleaning | Shape changed from {before_shape} to {self.df.shape}"
        )
        return self

    # -------------------------------------------------------------------------
    # 4. RUN FULL PIPELINE
    # -------------------------------------------------------------------------
    def run_cleaning_pipeline(
        self,
        drop_cols: list[str] = None,
        keep_datetime_cols: list[str] = None,
        keep_response_cols: list[str] = None,
        keep_explanatory_cols: list[str] = None,
        derivation_map: dict = None,
        drop_components: dict[str, list[str]] = None,
        categorical_config: dict = None,
        min_minority_count: int = 10,
        max_missing_rate: float = 0.40,
    ):
        """
        Run all cleaning steps in sequence.

        Parameters:
            drop_cols: List of columns to drop immediately without any checks
            keep_datetime_cols: List of datetime columns to keep. If None, all datetime columns are dropped.
            keep_response_cols: List of response/outcome columns to keep
            keep_explanatory_cols: List of explanatory/predictor columns to keep
            derivation_map: Dictionary of new variables to derive
            drop_components: Dictionary mapping derived columns to component columns to drop
            categorical_config: Configuration for categorical cleaning
            min_minority_count: Minimum observations in second largest class (default 10)
            max_missing_rate: Maximum proportion of missing data allowed (default 0.40)
        """
        self._log("=== Starting Dataset Cleaning Pipeline ===")

        # Drop specified columns first
        if drop_cols is not None:
            self.drop_columns(drop_cols)

        # Always run datetime column filtering (drops all by default if keep_datetime_cols not specified)
        self.keep_datetime_columns(keep_datetime_cols)

        if keep_response_cols is not None:
            self.keep_response_columns(keep_response_cols)

        if keep_explanatory_cols is not None:
            self.keep_explanatory_columns(keep_explanatory_cols)

        if derivation_map:
            self.derive_variables(derivation_map, drop_components=drop_components)

        if categorical_config:
            self.clean_categorical_predictors(
                consolidation_maps=categorical_config.get("consolidation_maps", {}),
                ordered_maps=categorical_config.get("ordered_maps", {}),
            )

        # Apply statistical power-based filtering
        self.drop_high_missingness_variables(max_missing_rate=max_missing_rate)
        self.drop_imbalanced_variables(min_minority_count=min_minority_count)

        self._log("=== Dataset Cleaning Complete ===")
        return self.df, self.log_messages
```

</details>

## Data Preparation

<details>
<summary>DataPreparer</summary>

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Optional, Tuple


class DataPreparer:
    """
    Prepares a dataset for modeling with flexible preprocessing options.
    """

    def __init__(
        self, df: pd.DataFrame, response_variable: str, explanatory_variables: list
    ):
        self.df = df.copy()
        self.response_variable = response_variable
        self.explanatory_variables = explanatory_variables
        # Placeholders for processed data
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None

    def check_separation(
        self, threshold: float = 0.95, verbose: bool = True
    ) -> list[str]:
        """
        Check for perfect or quasi-perfect separation between predictors and outcome.
        Should be called BEFORE preprocessing to check raw variables.
        Ignores NaN values in predictors since they'll be dropped during preprocessing.

        Parameters
        ----------
        threshold : float
            Proportion threshold for quasi-separation (default 0.95).
            If any predictor level has ≥95% observations in one outcome class, flag it.
        verbose : bool
            Whether to print detailed separation information.

        Returns
        -------
        list[str]
            List of predictor names with separation issues that should be excluded.
        """
        problematic = []
        outcome = self.df[self.response_variable]

        for pred in self.explanatory_variables:
            pred_data = self.df[pred]

            # For categorical/integer predictors, check cross-tabulation
            if pred_data.dtype in [
                "object",
                "category",
                "int64",
                "Int64",
                "int32",
                "int16",
                "int8",
            ]:
                # Use dropna=False to see NaN but we'll skip checking it
                crosstab = pd.crosstab(pred_data, outcome, dropna=False)

                # Check each level of the predictor (excluding NaN)
                for idx in crosstab.index:
                    # Skip NaN index
                    if pd.isna(idx):
                        continue

                    row = crosstab.loc[idx]
                    if row.sum() == 0:  # Empty row
                        continue
                    if len(row[row > 0]) < 2:  # Only one outcome level present
                        problematic.append(pred)
                        if verbose:
                            print(
                                f"Perfect separation in '{pred}': level '{idx}' has 100% in one outcome"
                            )
                        break

                    # Calculate proportion in dominant outcome
                    max_prop = row.max() / row.sum()

                    if max_prop >= threshold:
                        problematic.append(pred)
                        if verbose:
                            dominant_outcome = row.idxmax()
                            print(
                                f"Quasi-separation in '{pred}': level '{idx}' has {max_prop*100:.1f}% in '{dominant_outcome}'"
                            )
                        break

        return list(set(problematic))  # Remove duplicates

    def preprocess(
        self,
        drop_first_category: bool = True,
        scale_numeric: bool = True,
        min_category_size: int = 5,
        check_separation: bool = True,
        separation_threshold: float = 0.95,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Processes the explanatory variables and separates the response.

        Parameters
        ----------
        drop_first_category : bool
            Whether to drop the first category when one-hot encoding.
        scale_numeric : bool
            Whether to standardize numeric variables.
        min_category_size : int
            Minimum number of observations per categorical class to retain it.
        check_separation : bool
            Whether to check for separation issues and warn (default True).
            Does NOT automatically drop - just alerts you.
        separation_threshold : float
            Proportion threshold for quasi-separation warning (default 0.95).
            Only used if check_separation is True.

        Returns
        -------
        X : pd.DataFrame
            Preprocessed feature matrix ready for modeling.
        y : pd.Series
            Response variable series.
        """
        # Check for separation and warn (but don't drop)
        if check_separation:
            print("Checking for separation issues...")
            problematic = self.check_separation(
                threshold=separation_threshold, verbose=True
            )
            if problematic:
                print(
                    f"\nWARNING: {len(problematic)} predictors may cause convergence issues in standard logistic regression."
                )
                print(
                    "    Consider using penalized regression (Firth, Ridge) or manually excluding these variables."
                )

        # Separate response and predictors
        y = self.df[self.response_variable].copy()
        X = self.df[self.explanatory_variables].copy()

        # Identify numeric vs categorical columns
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

        # Filter rare categories in categorical variables
        for col in categorical_cols:
            # Future-proof categorical check
            if not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")

            counts = X[col].value_counts()
            rare_labels = counts[counts < min_category_size].index
            if len(rare_labels) > 0:
                X[col] = X[col].cat.remove_categories(rare_labels)

            # Drop rows that now have NaN after removing rare categories
            X = X[X[col].notna()]
            y = y.loc[X.index]

        # Drop any remaining rows with missing response or predictors
        X = X.dropna()
        y = y.loc[X.index]

        # Scale numeric columns if requested
        if scale_numeric and numeric_cols:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # One-hot encode categorical variables if any
        if categorical_cols:
            encoder = OneHotEncoder(
                drop="first" if drop_first_category else None,
                sparse_output=False,
                dtype=int,
            )
            encoded = encoder.fit_transform(X[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(categorical_cols),
                index=X.index,
            )
            X = X.drop(columns=categorical_cols)
            X = pd.concat([X, encoded_df], axis=1)

        self.X = X
        self.y = y
        return self.X, self.y
```

</details>

## Variable Selection

<details>
<summary>UnivariateVariableSelection</summary>

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Union, Optional
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from pandas.api.types import CategoricalDtype, is_object_dtype, is_numeric_dtype

from .preparation import DataPreparer


class UnivariateVariableSelection:
    """
    Perform univariate statistical tests between a response variable and a set of explanatory variables.

    Supports:
        - Continuous vs continuous: Pearson correlation
        - Continuous vs categorical: t-test (binary) or ANOVA (multi-class)
        - Categorical vs categorical: Chi-square or Fisher's exact (2x2 small counts)
        - Categorical vs continuous: t-test (binary) or ANOVA (multi-class)

    Attributes:
        df (pd.DataFrame): Subset dataframe containing only response and explanatory variables.
        response_variable (str): Name of the response variable.
        explanatory_variables (List[str]): List of candidate explanatory variables.
        alpha (float): Significance level for variable selection.
        results_df (pd.DataFrame): Results of univariate testing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        response_variable: str,
        explanatory_variables: List[str],
        alpha: float = 0.1,
    ):
        self.response_variable = response_variable
        self.explanatory_variables = explanatory_variables
        self.columns = [response_variable] + explanatory_variables
        self.alpha = alpha
        self.df = self._subset_dataframe(df)
        self.n = self.df.shape[0]
        self.p = self.df.shape[1] - 1
        self.results_df = self._perform_univariate_selection()
        print(
            f"\nSelected {len(self.selected_explanatory_variables)} of {len(self.explanatory_variables)} variables (p < {self.alpha})"
        )

    def _subset_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subset the dataframe to remove missing response variable values."""
        df = df.copy()[~df[self.response_variable].isnull()][self.columns]
        print(f"\n{self.response_variable:<60}{str(df.shape):<10}")
        return df

    @property
    def shape(self):
        return self.df.shape

    @property
    def selected_explanatory_variables(self) -> List[str]:
        """Return list of variables with p-value below alpha."""
        return self.results_df[self.results_df["p_value"] < self.alpha][
            "variable"
        ].to_list()

    def get_col_scale(self, column: str) -> str:
        """Public wrapper for column type: continuous or categorical."""
        return self._get_col_scale(column)

    def _get_col_scale(self, column: str) -> str:
        """Determine if a column is continuous or categorical."""
        dtype = self.df[column].dtype
        if isinstance(dtype, CategoricalDtype) or is_object_dtype(self.df[column]):
            return "categorical"
        elif is_numeric_dtype(self.df[column]):
            return "continuous"
        else:
            raise ValueError(
                f"Column '{column}' must be numeric or categorical, got {dtype}"
            )

    def _perform_univariate_selection(self) -> pd.DataFrame:
        results = []
        y = self.df[self.response_variable]
        y_type = self._get_col_scale(self.response_variable)

        for var in self.explanatory_variables:
            x = self.df[var]
            x_type = self._get_col_scale(var)

            if x.nunique() < 2:
                print(f"'{var}' has less than 2 unique values, skipping.")
                continue

            test_name, p_val, stat = self._run_test(x, x_type, y, y_type)
            if p_val is not None:
                results.append(
                    {
                        "variable": var,
                        "test": test_name,
                        "p_value": p_val,
                        "statistic": stat,
                    }
                )

        return pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)

    def _run_test(self, x, x_type: str, y, y_type: str) -> Union[str, float, float]:
        """Run appropriate statistical test for the variable pair."""
        stat = None

        # Continuous outcome
        if y_type == "continuous":
            if x_type == "continuous":
                r, p_val = self._pearson(x, y)
                test_name = "pearson"
                stat = r
            else:
                groups = [y[x == lvl].dropna() for lvl in x.unique()]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    print(f"'{x.name}' has insufficient non-missing data, skipping.")
                    return None, None, None
                if len(groups) == 2:
                    stat, p_val = self._t_test(groups)
                    test_name = "t-test"
                else:
                    stat, p_val = self._anova(groups)
                    test_name = "ANOVA"

        # Categorical outcome
        else:
            if x_type == "categorical":
                table = pd.crosstab(x, y)
                if table.shape == (2, 2) and (table.values < 5).any():
                    stat, p_val = self._fisher(table)
                    test_name = "fisher"
                else:
                    chi_table = self._filter_crosstab(
                        pd.crosstab(x, y, margins=True, margins_name="class_count")
                    )
                    if chi_table.shape[0] <= 1 or chi_table.shape[1] <= 1:
                        return None, None, None
                    stat, p_val, _, _ = self._chi_square(chi_table)
                    test_name = "chi-square"
            else:
                groups = [x[y == lvl].dropna() for lvl in y.unique()]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    return None, None, None
                if len(groups) == 2:
                    stat, p_val = self._t_test(groups)
                    test_name = "t-test"
                else:
                    stat, p_val = self._anova(groups)
                    test_name = "ANOVA"

        return test_name, p_val, stat

    def _filter_crosstab(self, table: pd.DataFrame, min_class_size=5) -> pd.DataFrame:
        """Remove rows/columns with class counts below threshold."""
        table = table.copy()
        table = table[table["class_count"] >= min_class_size]
        table = table.loc[:, table.loc["class_count"] >= min_class_size]
        if table.shape[0] <= 1 or table.shape[1] <= 1:
            return table
        return table.iloc[:-1, :-1]

    # Statistical test wrappers
    def _pearson(self, x, y):
        return stats.pearsonr(x, y)

    def _t_test(self, groups: list):
        return stats.ttest_ind(groups[0], groups[1], equal_var=False)

    def _anova(self, groups: list):
        return stats.f_oneway(*groups)

    def _fisher(self, table: pd.DataFrame):
        if table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires a 2x2 table")
        return stats.fisher_exact(table)

    def _chi_square(self, table: pd.DataFrame):
        return stats.chi2_contingency(table)
```

</details>

<details>
<summary>ElasticNetVariableSelection</summary>

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Union, Optional
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from pandas.api.types import CategoricalDtype, is_object_dtype, is_numeric_dtype

from .preparation import DataPreparer


class ElasticNetVariableSelection:
    """
    Perform variable selection using Elastic Net regularization.

    Supports:
    - Binary outcomes: Logistic regression with elastic net penalty
    - Count outcomes: Elastic net regression (Poisson-like)

    Variables with non-zero coefficients after regularization are selected.

    Attributes:
        df (pd.DataFrame): Subset dataframe containing only response and explanatory variables.
        response_variable (str): Name of the response variable.
        explanatory_variables (List[str]): List of candidate explanatory variables.
        outcome_type (str): Type of outcome - 'binary' or 'count'.
        alpha_ratio (float): L1 ratio for elastic net (0=Ridge, 1=Lasso, 0.5=Elastic Net).
        cv (int): Number of cross-validation folds.
        selected_explanatory_variables (List[str]): Variables selected by elastic net.
        results_df (pd.DataFrame): Results showing coefficient for each variable.
        model: Fitted elastic net model.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        response_variable: str,
        explanatory_variables: List[str],
        outcome_type: str = "binary",
        alpha_ratio: float = 0.5,
        cv: int = 5,
    ):
        """
        Initialize and perform elastic net variable selection.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing response and explanatory variables.
        response_variable : str
            Name of the response variable.
        explanatory_variables : List[str]
            List of candidate predictor variable names.
        outcome_type : str
            Type of outcome: 'binary' for binary outcomes, 'count' for count outcomes.
        alpha_ratio : float
            L1 ratio for elastic net (0 = Ridge, 1 = Lasso, 0.5 = Elastic Net).
            Default is 0.5 (balanced elastic net).
        cv : int
            Number of cross-validation folds. Default is 5.
        """
        self.df = df[[response_variable] + explanatory_variables].copy()
        self.response_variable = response_variable
        self.explanatory_variables = explanatory_variables
        self.outcome_type = outcome_type
        self.alpha_ratio = alpha_ratio
        self.cv = cv

        self.selected_explanatory_variables: List[str] = []
        self.results_df: Optional[pd.DataFrame] = None
        self.model = None
        self.X_prepared = None
        self.y_prepared = None

        # Perform selection
        self._prepare_data()
        self._fit_elastic_net()
        self._extract_selected_variables()

    def _prepare_data(self):
        """Use DataPreparer to prepare data for elastic net."""
        preparer = DataPreparer(
            df=self.df,
            response_variable=self.response_variable,
            explanatory_variables=self.explanatory_variables,
        )

        # Prepare data (scales numeric, one-hot encodes categorical, drops missing)
        self.X_prepared, self.y_prepared = preparer.preprocess(
            drop_first_category=True,
            scale_numeric=True,  # Elastic net needs scaling
            min_category_size=1,  # Don't drop rare categories here - let elastic net handle
            check_separation=False,  # Don't need separation check for elastic net
        )

    def _fit_elastic_net(self):
        """Fit elastic net model with cross-validation."""
        if self.outcome_type == "binary":
            # Logistic regression with elastic net
            self.model = LogisticRegressionCV(
                penalty="elasticnet",
                solver="saga",
                l1_ratios=[self.alpha_ratio],
                cv=self.cv,
                max_iter=10000,
                random_state=42,
                n_jobs=-1,
            )
        elif self.outcome_type == "count":
            # Elastic net for count data
            self.model = ElasticNetCV(
                l1_ratio=self.alpha_ratio,
                cv=self.cv,
                max_iter=10000,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(
                f"outcome_type must be 'binary' or 'count', got '{self.outcome_type}'"
            )

        self.model.fit(self.X_prepared, self.y_prepared)

    def _extract_selected_variables(self):
        """Extract variables with non-zero coefficients and create results dataframe."""
        # Get coefficients
        if self.outcome_type == "binary":
            coefs = self.model.coef_[0]
        else:
            coefs = self.model.coef_

        # Create results dataframe
        self.results_df = pd.DataFrame(
            {
                "Variable": self.X_prepared.columns,
                "Coefficient": coefs,
                "Abs_Coefficient": np.abs(coefs),
                "Selected": np.abs(coefs) > 1e-5,
            }
        ).sort_values("Abs_Coefficient", ascending=False)

        # Get selected encoded features
        selected_encoded = self.results_df[self.results_df["Selected"]][
            "Variable"
        ].tolist()

        # Map back to original variable names
        original_vars = set()
        for encoded_var in selected_encoded:
            # Check if it's a one-hot encoded variable (has underscore from OneHotEncoder)
            found = False
            for orig_var in self.explanatory_variables:
                if encoded_var.startswith(orig_var + "_"):
                    original_vars.add(orig_var)
                    found = True
                    break

            # If not found, it's a numeric variable (not one-hot encoded)
            if not found and encoded_var in self.explanatory_variables:
                original_vars.add(encoded_var)

        self.selected_explanatory_variables = sorted(list(original_vars))

        print(
            f"Elastic Net selected {len(self.selected_explanatory_variables)} of {len(self.explanatory_variables)} variables"
        )
        if len(self.selected_explanatory_variables) > 0:
            print(f"Selected variables: {self.selected_explanatory_variables}")
        else:
            print("WARNING: No variables selected - regularization may be too strong")
```

</details>

## Statistical Modeling

<details>
<summary>BaseModel</summary>

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.discrete.discrete_model import NegativeBinomial
from typing import Optional


class BaseModel:
    """
    Base class for regression models, storing X, y, and fitted model.
    """

    def __init__(self):
        self.model = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError("Subclasses must implement this method.")

    def summary(self):
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        return self.model.summary()

    def check_convergence(self):
        """Check if the model converged and print warning if not."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")

        if hasattr(self.model, "mle_retvals") and "converged" in self.model.mle_retvals:
            self.converged = self.model.mle_retvals["converged"]
            if not self.converged:
                print("WARNING: Maximum Likelihood optimization failed to converge.")
                print(
                    "    Results may be unreliable. Check coefficients and standard errors."
                )
                return False
        return True
```

</details>

<details>
<summary>LogisticRegressionModel</summary>

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.discrete.discrete_model import NegativeBinomial
from typing import Optional


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X, self.y = X.copy(), y.copy()

        # Ensure numeric
        self.X = self.X.astype(float)
        if self.y.dtype.name == "category" or self.y.dtype == object:
            self.y = (
                self.y.cat.codes
                if hasattr(self.y, "cat")
                else self.y.map({self.y.unique()[0]: 0, self.y.unique()[1]: 1})
            )
        self.y = self.y.astype(int)

        # Add constant for intercept
        X_sm = sm.add_constant(self.X, has_constant="add")

        # Fit logistic regression
        self.model = sm.Logit(self.y, X_sm).fit(disp=False)

        # Check convergence
        self.check_convergence()

        return self.model
```

</details>

<details>
<summary>NegativeBinomialRegressionModel</summary>

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.discrete.discrete_model import NegativeBinomial
from typing import Optional


class NegativeBinomialRegressionModel(BaseModel):
    """
    Fits a Negative Binomial regression model for count outcomes.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X, self.y = X, y
        # Add constant term for intercept
        X_sm = sm.add_constant(self.X, has_constant="add")

        # Fit negative binomial regression
        self.model = NegativeBinomial(self.y, X_sm).fit(disp=False)

        # Check convergence
        self.check_convergence()

        return self.model
```

</details>
