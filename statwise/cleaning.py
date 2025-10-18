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
