import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Optional, Tuple


class DataPreparer:
    """
    Prepare cleaned datasets for statistical modeling with flexible preprocessing options.

    This class transforms cleaned data into model-ready format through standardization,
    encoding, and handling of rare categories. It checks for separation issues that
    can cause convergence problems in logistic regression.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame containing response and explanatory variables.
    response_variable : str
        Name of the response/outcome variable column.
    explanatory_variables : list of str
        List of predictor variable column names to include in modeling.

    Attributes
    ----------
    df : pd.DataFrame
        Working copy of the DataFrame.
    response_variable : str
        Response variable name.
    explanatory_variables : list of str
        Explanatory variable names.
    X : pd.DataFrame or None
        Preprocessed feature matrix (None until preprocess() is called).
    y : pd.Series or None
        Response variable series (None until preprocess() is called).

    Examples
    --------
    >>> preparer = DataPreparer(
    ...     df=cleaned_df,
    ...     response_variable='SSI',
    ...     explanatory_variables=['Age', 'BMI', 'Diabetes', 'INPWT']
    ... )
    >>> X, y = preparer.preprocess(
    ...     drop_first_category=True,
    ...     scale_numeric=True,
    ...     min_category_size=5
    ... )

    See Also
    --------
    DataCleaner : Cleans raw data before preparation.
    LogisticRegressionModel : Fits logistic models using prepared data.
    NegativeBinomialRegressionModel : Fits count models using prepared data.

    Notes
    -----
    Preprocessing operations performed:
    1. Separation checking (warns about convergence risks)
    2. Rare category filtering (removes categories with <5 observations)
    3. Missing data removal (complete case analysis)
    4. Numeric standardization (z-score transformation)
    5. Categorical encoding (one-hot encoding with optional reference category)

    The class does not perform imputation - missing values result in row removal.
    For datasets with substantial missingness, consider imputation before this step.
    """

    def __init__(
        self, df: pd.DataFrame, response_variable: str, explanatory_variables: list
    ):
        self.df = df.copy()
        self.response_variable = response_variable
        self.explanatory_variables = explanatory_variables
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None

    def check_separation(
        self, threshold: float = 0.95, verbose: bool = True
    ) -> list[str]:
        """
        Check for perfect or quasi-perfect separation between predictors and outcome.

        Separation occurs when a predictor (or combination) perfectly or nearly
        perfectly predicts the outcome, causing maximum likelihood estimation to
        fail or produce unreliable results with infinite standard errors.

        Parameters
        ----------
        threshold : float, default 0.95
            Proportion threshold for quasi-separation. If any predictor level
            has ≥95% observations in one outcome class, it is flagged.
        verbose : bool, default True
            Whether to print detailed separation information.

        Returns
        -------
        list of str
            Names of predictors with separation issues that may cause convergence
            problems and should be considered for exclusion or regularization.

        Examples
        --------
        >>> preparer = DataPreparer(df, 'SSI', ['Age', 'Rare_Complication'])
        >>> problematic = preparer.check_separation(threshold=0.95, verbose=True)
        Perfect separation in 'Rare_Complication': level 'Yes' has 100% in one outcome
        >>> print(problematic)
        ['Rare_Complication']

        Notes
        -----
        Types of separation:
        - **Perfect separation**: A predictor level has 100% of observations in
          one outcome class (e.g., all patients with rare complication died)
        - **Quasi-separation**: A predictor level has ≥95% in one outcome class
          (e.g., 48 of 50 patients with condition had the outcome)

        NaN values in predictors are ignored during separation checking since they
        will be dropped during preprocessing.

        For predictors flagged with separation:
        - Consider using penalized regression (Firth, Ridge, Lasso)
        - Exclude the variable if not theoretically important
        - Combine categories to reduce separation
        - Use exact logistic regression for small samples

        References
        ----------
        .. [1] Heinze G, Schemper M. "A solution to the problem of separation in
               logistic regression." Stat Med. 2002;21(16):2409-19.
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
                crosstab = pd.crosstab(pred_data, outcome, dropna=False)

                # Check each level of the predictor (excluding NaN)
                for idx in crosstab.index:
                    if pd.isna(idx):
                        continue

                    row = crosstab.loc[idx]
                    if row.sum() == 0:
                        continue

                    if len(row[row > 0]) < 2:
                        problematic.append(pred)
                        if verbose:
                            print(
                                f"Perfect separation in '{pred}': level '{idx}' has 100% in one outcome"
                            )
                        break

                    max_prop = row.max() / row.sum()

                    if max_prop >= threshold:
                        problematic.append(pred)
                        if verbose:
                            dominant_outcome = row.idxmax()
                            print(
                                f"Quasi-separation in '{pred}': level '{idx}' has {max_prop*100:.1f}% in '{dominant_outcome}'"
                            )
                        break

        return list(set(problematic))

    def preprocess(
        self,
        drop_first_category: bool = True,
        scale_numeric: bool = True,
        min_category_size: int = 5,
        check_separation: bool = True,
        separation_threshold: float = 0.95,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process explanatory variables and response into model-ready format.

        This method performs all preprocessing steps needed to prepare data for
        statistical modeling, including separation checking, rare category handling,
        standardization, and encoding.

        Parameters
        ----------
        drop_first_category : bool, default True
            Whether to drop the first category when one-hot encoding categorical
            variables. True creates a reference category and avoids perfect
            multicollinearity. Set to False for models that handle multicollinearity
            (e.g., tree-based methods).
        scale_numeric : bool, default True
            Whether to standardize numeric variables to mean=0, std=1.
            Recommended for regression with regularization (elastic net, ridge).
        min_category_size : int, default 5
            Minimum number of observations required for a categorical level to
            be retained. Categories with fewer observations are removed, and
            those rows are excluded from analysis.
        check_separation : bool, default True
            Whether to check for separation issues and print warnings.
            Does NOT automatically drop variables - just alerts for manual review.
        separation_threshold : float, default 0.95
            Proportion threshold for quasi-separation warning.
            Only used if check_separation is True.

        Returns
        -------
        X : pd.DataFrame
            Preprocessed feature matrix ready for modeling.
            - Numeric columns are standardized (if scale_numeric=True)
            - Categorical columns are one-hot encoded
            - All missing values removed (complete cases only)
            - Column names use format: original_name or categorical_level for encoded variables
        y : pd.Series
            Response variable series aligned with X (same index after missing data removal).

        Examples
        --------
        >>> preparer = DataPreparer(df, 'SSI', ['Age', 'Sex', 'BMI', 'INPWT'])
        >>> X, y = preparer.preprocess()
        Checking for separation issues...
        >>> print(X.shape, y.shape)
        (150, 5) (150,)
        >>> print(X.columns)
        Index(['Age', 'BMI', 'Sex_Male', 'INPWT_Yes'], dtype='object')

        >>> # For elastic net (needs scaling)
        >>> X, y = preparer.preprocess(scale_numeric=True, drop_first_category=True)

        >>> # For decision trees (no scaling or reference category needed)
        >>> X, y = preparer.preprocess(scale_numeric=False, drop_first_category=False)

        Notes
        -----
        Processing order:
        1. Separation checking (if enabled)
        2. Data subsetting (select X and y)
        3. Rare category removal (categories with <min_category_size)
        4. Missing data removal (complete case analysis)
        5. Numeric standardization (if enabled)
        6. Categorical one-hot encoding
        7. Int64 to float64 conversion (for statsmodels compatibility)

        The method uses scikit-learn's StandardScaler for numeric scaling and
        OneHotEncoder for categorical encoding, ensuring consistency with
        scikit-learn model pipelines.

        Warnings
        --------
        If separation is detected, a warning is printed but variables are not
        automatically removed. Consider using penalized regression or manually
        excluding problematic variables.
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

        # Convert nullable Int64 to regular float64 for statsmodels compatibility
        for col in X.columns:
            if X[col].dtype == "Int64":
                X[col] = X[col].astype("float64")

        if y.dtype == "Int64":
            y = y.astype("float64")

        self.X = X
        self.y = y
        return self.X, self.y
