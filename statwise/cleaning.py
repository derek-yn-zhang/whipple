import pandas as pd


class DataCleaner:
    """
    Modular data cleaning pipeline for reproducible preprocessing of surgical outcomes data.

    This class provides a structured approach to data cleaning with explicit,
    user-controlled operations that ensure reproducibility. All operations are
    logged and can be applied in a consistent order through the pipeline interface.

    Core functionality:
    - Remove irrelevant or unusable columns
    - Filter datetime and outcome variables
    - Drop statistically problematic variables (imbalanced, high missingness)
    - Derive composite or standardized variables
    - Clean and order categorical predictors

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to clean.
    column_metadata : dict of {str: str}
        Dictionary mapping column names to their role in analysis.
        Valid values are 'response' (outcome variables) or 'explanatory' (predictors).
    log : bool, default True
        Whether to print log messages during cleaning operations.

    Attributes
    ----------
    df : pd.DataFrame
        Working copy of the DataFrame being cleaned.
    column_metadata : dict
        Column role metadata for identifying response vs explanatory variables.
    log : bool
        Whether logging is enabled.
    log_messages : list of str
        Accumulated log messages from all cleaning operations.

    Examples
    --------
    >>> metadata = {
    ...     'SSI': 'response',
    ...     'Sepsis': 'response',
    ...     'Age': 'explanatory',
    ...     'BMI': 'explanatory',
    ...     'INPWT': 'explanatory'
    ... }
    >>> cleaner = DataCleaner(df, metadata)
    >>> cleaner.drop_columns(['Patient_ID', 'MRN'])
    >>> cleaner.drop_imbalanced_variables(min_minority_count=10)
    >>> cleaned_df, logs = cleaner.df, cleaner.log_messages

    See Also
    --------
    DataPreparer : Prepares cleaned data for modeling.
    CSVDataLoader : Loads CSV data with type inference.

    Notes
    -----
    The cleaning pipeline follows best practices for clinical research:
    - Removes variables with insufficient statistical power (10 events per variable rule)
    - Handles missing data conservatively (drops variables >40% missing by default)
    - Distinguishes between categorical and continuous integer variables
    - Preserves data provenance through detailed logging

    References
    ----------
    .. [1] Peduzzi P, Concato J, Kemper E, et al. "A simulation study of the
           number of events per variable in logistic regression analysis."
           J Clin Epidemiol. 1996;49(12):1373-9.
    """

    def __init__(
        self, df: pd.DataFrame, column_metadata: dict[str, str], log: bool = True
    ):
        self.df = df.copy()
        self.column_metadata = column_metadata
        self.log = log
        self.log_messages = []

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

    # -------------------------------------------------------------------------
    # 1. DROPPING METHODS
    # -------------------------------------------------------------------------
    def drop_columns(self, columns: list[str]):
        """
        Drop specified columns immediately without any checks or conditions.

        This method is useful for removing known unusable or irrelevant columns
        upfront (e.g., patient identifiers, administrative fields, redundant dates).

        Parameters
        ----------
        columns : list of str
            List of column names to drop. Non-existent columns are silently ignored.

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining.

        Examples
        --------
        >>> cleaner.drop_columns(['Patient_ID', 'MRN', 'Attending_Surgeon'])
        Dropping 3 specified columns: ['Patient_ID', 'MRN', 'Attending_Surgeon']
        Shape changed from (165, 45) to (165, 42)
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
        Keep only specified datetime columns, dropping all other datetime columns.

        If no columns are specified, drops all datetime columns. This is useful
        because datetime variables typically cannot be directly used as predictors
        in regression models and may indicate data linkage issues if present.

        Parameters
        ----------
        columns : list of str or None, default None
            List of datetime column names to keep. If None, drops all datetime columns.

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining.

        Examples
        --------
        >>> # Drop all datetime columns
        >>> cleaner.keep_datetime_columns()

        >>> # Keep only surgery date
        >>> cleaner.keep_datetime_columns(['Surgery_Date'])

        Notes
        -----
        Auto-detects datetime columns by checking for datetime64, datetime64[ns],
        or datetimetz dtypes.
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

        Uses column_metadata to identify which columns are response variables,
        then retains only those specified in the columns parameter.

        Parameters
        ----------
        columns : list of str
            List of response column names to keep.

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining.

        Examples
        --------
        >>> # Keep only SSI and sepsis outcomes, drop other complications
        >>> cleaner.keep_response_columns(['SSI', 'Sepsis'])

        Notes
        -----
        Response columns are identified from column_metadata where the value is 'response'.
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

        Uses column_metadata to identify which columns are explanatory variables,
        then retains only those specified in the columns parameter.

        Parameters
        ----------
        columns : list of str
            List of explanatory column names to keep.

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining.

        Examples
        --------
        >>> # Keep only demographic and treatment variables
        >>> cleaner.keep_explanatory_columns(['Age', 'Sex', 'BMI', 'INPWT'])

        Notes
        -----
        Explanatory columns are identified from column_metadata where the value
        is 'explanatory'.
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
        max_absolute_unique: int = 10,
    ) -> bool:
        """
        Determine if an integer column behaves categorically using multiple criteria.

        Treats a column as categorical if EITHER:
        1. Has ≤10 unique values (regardless of sample size), OR
        2. Unique/total ratio < 10%

        This approach correctly identifies:
        - Binary/ordinal scales (0-5 ratings) → always categorical
        - Low-cardinality counts in large samples → categorical
        - High-cardinality true counts → continuous

        Parameters
        ----------
        series : pd.Series
            Pandas Series to evaluate.
        uniqueness_threshold : float, default 0.10
            Maximum uniqueness ratio to treat as categorical.
        max_absolute_unique : int, default 10
            Maximum unique values to always treat as categorical.

        Returns
        -------
        bool
            True if should be treated as categorical, False if continuous.

        Examples
        --------
        >>> # Binary variable: 2 unique values → categorical
        >>> _is_quasi_categorical(pd.Series([0, 1, 0, 1, 0]))
        True

        >>> # Ordinal scale: 5 unique values → categorical
        >>> _is_quasi_categorical(pd.Series([1, 2, 3, 4, 5, 3, 2, 1]))
        True

        >>> # True count variable: 50 unique values in 200 obs → continuous
        >>> _is_quasi_categorical(pd.Series(range(200)))
        False

        Notes
        -----
        This heuristic is important for clinical data where integer columns may
        represent either categorical scales (ASA class: 1-5) or true counts
        (length of stay: 0-100+ days). The distinction affects whether balance
        checking and other categorical-specific operations should apply.
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
        max_absolute_unique: int = 10,
    ):
        """
        Drop categorical variables with insufficient observations in the second largest class.

        Uses the second largest class (not smallest) to allow variables with long tails
        of rare events, which can be merged during analysis. This is important for
        clinical data where rare complications may exist but the main contrast
        (e.g., None vs Any) is still meaningful.

        Integer columns are only checked if they behave categorically based on:
        - Having ≤10 unique values (regardless of sample size), OR
        - Having <10% unique-to-total ratio

        Parameters
        ----------
        min_minority_count : int, default 10
            Minimum observations required in the second largest class.
            Default of 10 follows the common "10 events per variable" rule
            for logistic regression [1]_.
        uniqueness_threshold : float, default 0.10
            Maximum uniqueness ratio for integers to be checked.
        max_absolute_unique : int, default 10
            Maximum unique values to always treat as categorical.

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining.

        Examples
        --------
        >>> cleaner.drop_imbalanced_variables(min_minority_count=10)
        Checking class imbalance for 15 categorical and quasi-categorical integer columns
        'Rare_Complication' (response): second largest class has 3 observations (< 10)
          Class counts: No: 162, Yes: 3
        Dropping 1 imbalanced variables with insufficient second class size

        Notes
        -----
        Variables with only one observed class are always dropped regardless of
        min_minority_count, as they provide no statistical information.

        References
        ----------
        .. [1] Peduzzi P, Concato J, Kemper E, et al. "A simulation study of the
               number of events per variable in logistic regression analysis."
               J Clin Epidemiol. 1996;49(12):1373-9.
        """
        before_shape = self.df.shape
        to_drop = []

        # Categorical columns always checked
        categorical_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Integer columns - filter to quasi-categorical only
        integer_cols = self.df.select_dtypes(
            include=["int64", "int32", "int16", "int8", "Int64"]
        ).columns.tolist()

        quasi_categorical_ints = [
            col
            for col in integer_cols
            if self._is_quasi_categorical(
                self.df[col],
                uniqueness_threshold=uniqueness_threshold,
                max_absolute_unique=max_absolute_unique,
            )
        ]

        # Track which integer columns are being skipped
        continuous_ints = [
            col for col in integer_cols if col not in quasi_categorical_ints
        ]

        cols_to_check = categorical_cols + quasi_categorical_ints

        self._log(
            f"Checking class imbalance for {len(cols_to_check)} categorical and quasi-categorical integer columns"
        )
        if continuous_ints:
            self._log(
                f"Skipping {len(continuous_ints)} continuous integer columns (high cardinality): {continuous_ints}"
            )

        for col in cols_to_check:
            value_counts = self.df[col].value_counts(dropna=True)

            if len(value_counts) == 0:
                continue

            if len(value_counts) == 1:
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
                counts_str = ", ".join(
                    [f"{val}: {count}" for val, count in value_counts.items()]
                )

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
        Drop variables with excessive missing data that would compromise statistical power.

        Applies to both response and explanatory variables. Variables exceeding the
        missingness threshold are removed because imputation becomes unreliable and
        complete case analysis would lose too much data.

        Parameters
        ----------
        max_missing_rate : float, default 0.40
            Maximum proportion of missing values allowed (0 to 1).
            Default of 0.40 (40%) is a common threshold in clinical research [1]_.

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining.

        Examples
        --------
        >>> cleaner.drop_high_missingness_variables(max_missing_rate=0.40)
        Checking missingness for 42 columns
        'Albumin' (explanatory): 65.5% missing (> 40%)
        Dropping 1 variables with excessive missingness

        Notes
        -----
        The 40% threshold represents a balance between preserving variables and
        maintaining data quality. Higher missingness rates make multiple imputation
        less reliable and reduce effective sample size substantially.

        References
        ----------
        .. [1] Jakobsen JC, Gluud C, Wetterslev J, Winkel P. "When and how should
               multiple imputation be used for handling missing data in randomised
               clinical trials - a practical guide with flowcharts."
               BMC Med Res Methodol. 2017;17(1):162.
        """
        before_shape = self.df.shape
        to_drop = []

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
        Create new variables from existing columns using user-supplied functions.

        Automatically infers metadata type (response/explanatory) for derived columns
        based on their component columns. Optionally drops component columns after
        derivation to avoid multicollinearity.

        Parameters
        ----------
        derivation_map : dict of {str: callable}
            Dictionary mapping new column names to derivation functions.
            Each function should take the full DataFrame as input and return a Series.
        drop_components : dict of {str: list of str}, optional
            Dictionary mapping new column names to lists of component columns to drop
            after derivation. If None, component columns are retained.

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining.

        Examples
        --------
        >>> def standardize_height(df):
        ...     # Convert all heights to cm
        ...     heights = df['Height'].copy()
        ...     heights[df['Height_Unit'] == 'in'] *= 2.54
        ...     return heights

        >>> def composite_sepsis(df):
        ...     # Any postoperative sepsis or septic shock
        ...     return ((df['Postop_Sepsis'] > 0) | (df['Postop_Septic_Shock'] > 0)).astype(int)

        >>> derivation_map = {
        ...     'Height_cm': standardize_height,
        ...     'Any_Postop_Sepsis': composite_sepsis
        ... }
        >>> drop_components = {
        ...     'Height_cm': ['Height', 'Height_Unit'],
        ...     'Any_Postop_Sepsis': ['Postop_Sepsis', 'Postop_Septic_Shock']
        ... }
        >>> cleaner.derive_variables(derivation_map, drop_components)

        Notes
        -----
        Metadata type inference rules:
        - If all component columns have the same type (all 'response' or all
          'explanatory'), the derived column inherits that type
        - If component columns have mixed types, defaults to 'explanatory'
        - If no component metadata found, defaults to 'explanatory'

        Common use cases in clinical data:
        - Standardizing measurements (height in cm, weight in kg)
        - Creating composite outcomes (any complication = SSI OR sepsis OR readmission)
        - Binary indicators from counts (any event = count > 0)
        - Risk scores from multiple predictors
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
                    unique_types = set(component_types)
                    if len(unique_types) == 1:
                        inferred_type = component_types[0]
                        self.column_metadata[new_col] = inferred_type
                        self._log(
                            f"  Inferred metadata type '{inferred_type}' for '{new_col}' based on component columns"
                        )
                    else:
                        self.column_metadata[new_col] = "explanatory"
                        self._log(
                            f"  Warning: Component columns have mixed types {unique_types}. Defaulting '{new_col}' to 'explanatory'"
                        )
                else:
                    self.column_metadata[new_col] = "explanatory"
                    self._log(
                        f"  No component metadata found. Defaulting '{new_col}' to 'explanatory'"
                    )
            else:
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

        This method enables two types of categorical cleaning:
        1. Consolidation: Merge or rename categories (e.g., combine rare races)
        2. Ordering: Apply ordinal relationships (e.g., ASA class I < II < III)

        Parameters
        ----------
        consolidation_maps : dict of {str: dict}
            Dictionary mapping column names to category consolidation mappings.
            Each mapping is a dict of {old_value: new_value}.
        ordered_maps : dict of {str: list}, optional
            Dictionary mapping column names to ordered lists of categories.
            Categories will be treated as ordinal in this order.

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining.

        Examples
        --------
        >>> consolidation_maps = {
        ...     'Race': {
        ...         'Black or African American': 'Black',
        ...         'White': 'White',
        ...         'Asian': 'Other',
        ...         'Some Other Race': 'Other'
        ...     },
        ...     'Smoking_Status': {
        ...         'Current': 'Current',
        ...         'Former': 'Former',
        ...         'Never': 'Never',
        ...         'Unknown': 'Never'  # Conservative assumption
        ...     }
        ... }
        >>> ordered_maps = {
        ...     'ASA_Class': [
        ...         'ASA I - Normal healthy',
        ...         'ASA II - Mild systemic disease',
        ...         'ASA III - Severe systemic disease',
        ...         'ASA IV - Severe disease, constant threat to life'
        ...     ]
        ... }
        >>> cleaner.clean_categorical_predictors(consolidation_maps, ordered_maps)

        Notes
        -----
        Consolidation is often necessary to:
        - Merge rare categories with insufficient sample size
        - Group conceptually similar categories
        - Handle data entry variations (e.g., "Hispanic" vs "Hispanic or Latino")

        Ordering is important for:
        - Ordinal scales (ASA class, pain scores, disease stages)
        - Enabling ordinal encoding in models that can leverage order
        - Correct interpretation of trends across ordered categories
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
        Execute all cleaning steps in a standardized sequence.

        This pipeline method provides a single entry point for reproducible data
        cleaning with explicit parameter documentation. The execution order is
        optimized to prevent conflicts between operations.

        Parameters
        ----------
        drop_cols : list of str, optional
            Columns to drop immediately (e.g., identifiers, administrative fields).
        keep_datetime_cols : list of str, optional
            Datetime columns to keep. If None, all datetime columns are dropped.
        keep_response_cols : list of str, optional
            Response/outcome columns to keep.
        keep_explanatory_cols : list of str, optional
            Explanatory/predictor columns to keep.
        derivation_map : dict of {str: callable}, optional
            New variables to derive from existing columns.
        drop_components : dict of {str: list of str}, optional
            Component columns to drop after deriving new variables.
        categorical_config : dict, optional
            Configuration for categorical cleaning with keys:
            - 'consolidation_maps': dict for category consolidation
            - 'ordered_maps': dict for applying ordinal structure
        min_minority_count : int, default 10
            Minimum observations in second largest class for balance check.
        max_missing_rate : float, default 0.40
            Maximum proportion of missing data allowed (0 to 1).

        Returns
        -------
        df : pd.DataFrame
            Cleaned DataFrame ready for preprocessing and modeling.
        log_messages : list of str
            Complete log of all cleaning operations performed.

        Examples
        --------
        >>> cleaner = DataCleaner(df, column_metadata)
        >>> cleaned_df, logs = cleaner.run_cleaning_pipeline(
        ...     drop_cols=['Patient_ID', 'MRN'],
        ...     keep_datetime_cols=None,  # Drop all datetime columns
        ...     keep_response_cols=['SSI', 'Sepsis', 'Length_of_Stay', 'Readmission'],
        ...     keep_explanatory_cols=['Age', 'Sex', 'BMI', 'Diabetes', 'INPWT'],
        ...     derivation_map={'Any_SSI': lambda df: (df['SSI'] > 0).astype(int)},
        ...     categorical_config={'consolidation_maps': {...}},
        ...     min_minority_count=10,
        ...     max_missing_rate=0.40
        ... )

        Notes
        -----
        Pipeline execution order:
        1. Drop specified columns
        2. Filter datetime columns (drops all by default)
        3. Filter response columns
        4. Filter explanatory columns
        5. Derive new variables
        6. Clean categorical variables
        7. Drop high-missingness variables
        8. Drop imbalanced variables

        This order ensures that:
        - Unwanted columns are removed early to improve clarity
        - Derived variables are available for subsequent operations
        - Statistical filtering happens after manual curation is complete
        """
        self._log("=== Starting Dataset Cleaning Pipeline ===")

        # Drop specified columns first
        if drop_cols is not None:
            self.drop_columns(drop_cols)

        # Always run datetime column filtering (drops all by default)
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
