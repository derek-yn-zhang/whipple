import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Union, Optional, Dict
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from pandas.api.types import CategoricalDtype, is_object_dtype, is_numeric_dtype
import statsmodels.api as sm

from .preparation import DataPreparer


class UnivariateVariableSelection:
    """
    Perform univariate statistical testing for variable selection.

    Tests the association between each explanatory variable and the response
    variable individually, using appropriate statistical tests based on variable
    types. Variables with p-values below the significance threshold are selected.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing response and explanatory variables.
    response_variable : str
        Name of the response/outcome variable.
    explanatory_variables : list of str
        List of candidate predictor variable names.
    alpha : float, default 0.1
        Significance level for variable selection. Variables with p < alpha
        are selected. Default of 0.1 is common for initial screening.

    Attributes
    ----------
    df : pd.DataFrame
        Subset DataFrame with response and explanatory variables (missing response removed).
    response_variable : str
        Response variable name.
    explanatory_variables : list of str
        Explanatory variable names.
    alpha : float
        Significance threshold for selection.
    n : int
        Number of observations after removing missing response values.
    p : int
        Number of explanatory variables.
    results_df : pd.DataFrame
        DataFrame containing test results for each variable, sorted by p-value.
    selected_explanatory_variables : list of str
        Variables with p-value < alpha.

    Examples
    --------
    >>> selector = UnivariateVariableSelection(
    ...     df=cleaned_df,
    ...     response_variable='SSI',
    ...     explanatory_variables=['Age', 'Sex', 'BMI', 'Diabetes', 'INPWT'],
    ...     alpha=0.1
    ... )
    Selected 3 of 5 variables (p < 0.1)

    >>> print(selector.results_df)
       variable       test   p_value  statistic
    0      INPWT  chi-square   0.023       5.17
    1   Diabetes  chi-square   0.067       3.36
    2        BMI    pearson   0.089       0.28

    >>> print(selector.selected_explanatory_variables)
    ['INPWT', 'Diabetes', 'BMI']

    See Also
    --------
    ElasticNetVariableSelection : Regularization-based variable selection.
    scipy.stats : Statistical functions used for testing.

    Notes
    -----
    Statistical tests applied based on variable types:

    +-------------------+-------------------+-------------------------+
    | Response Type     | Predictor Type    | Test                    |
    +===================+===================+=========================+
    | Continuous        | Continuous        | Pearson correlation     |
    +-------------------+-------------------+-------------------------+
    | Continuous        | Binary            | Independent t-test      |
    +-------------------+-------------------+-------------------------+
    | Continuous        | Multi-category    | One-way ANOVA           |
    +-------------------+-------------------+-------------------------+
    | Binary/Categorical| Continuous        | Independent t-test      |
    |                   |                   | or ANOVA                |
    +-------------------+-------------------+-------------------------+
    | Binary/Categorical| Binary/Categorical| Chi-square or           |
    |                   |                   | Fisher's exact (2x2)    |
    +-------------------+-------------------+-------------------------+

    Univariate selection limitations:
    - Does not account for confounding or interaction effects
    - May select collinear variables
    - Multiple testing increases false positive risk
    - Should be combined with other selection methods

    The default alpha=0.1 (rather than 0.05) is intentional for initial
    screening, as overly stringent thresholds may miss important confounders.

    References
    ----------
    .. [1] Bursac Z, Gauss CH, Williams DK, Hosmer DW. "Purposeful selection of
           variables in logistic regression." Source Code Biol Med. 2008;3:17.
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
        """
        Subset DataFrame and remove rows with missing response values.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Subset containing only relevant columns with non-missing response.
        """
        df = df.copy()[~df[self.response_variable].isnull()][self.columns]
        print(f"\n{self.response_variable:<60}{str(df.shape):<10}")
        return df

    @property
    def shape(self):
        """
        Return shape of the analysis DataFrame.

        Returns
        -------
        tuple of int
            (n_rows, n_columns) of the DataFrame.
        """
        return self.df.shape

    @property
    def selected_explanatory_variables(self) -> List[str]:
        """
        Return list of variables selected by univariate testing.

        Returns
        -------
        list of str
            Variables with p-value below the alpha threshold.
        """
        return self.results_df[self.results_df["p_value"] < self.alpha][
            "variable"
        ].to_list()

    def get_col_scale(self, column: str) -> str:
        """
        Determine if a column is continuous or categorical.

        Parameters
        ----------
        column : str
            Column name to evaluate.

        Returns
        -------
        str
            Either 'continuous' or 'categorical'.

        Raises
        ------
        ValueError
            If column is neither numeric nor categorical.
        """
        return self._get_col_scale(column)

    def _get_col_scale(self, column: str) -> str:
        """
        Internal method to determine column type.

        Parameters
        ----------
        column : str
            Column name.

        Returns
        -------
        str
            'continuous' or 'categorical'.
        """
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
        """
        Execute univariate tests for all explanatory variables.

        Returns
        -------
        pd.DataFrame
            Results with columns: variable, test, p_value, statistic.
            Sorted by p_value (ascending).
        """
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
        """
        Run appropriate statistical test based on variable types.

        Parameters
        ----------
        x : pd.Series
            Explanatory variable.
        x_type : str
            Type of x ('continuous' or 'categorical').
        y : pd.Series
            Response variable.
        y_type : str
            Type of y ('continuous' or 'categorical').

        Returns
        -------
        test_name : str or None
            Name of the test performed.
        p_val : float or None
            P-value from the test.
        stat : float or None
            Test statistic value.
        """
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
        """
        Remove rows/columns with class counts below threshold.

        Parameters
        ----------
        table : pd.DataFrame
            Crosstab with margins.
        min_class_size : int, default 5
            Minimum class size to retain.

        Returns
        -------
        pd.DataFrame
            Filtered crosstab.
        """
        table = table.copy()
        table = table[table["class_count"] >= min_class_size]
        table = table.loc[:, table.loc["class_count"] >= min_class_size]
        if table.shape[0] <= 1 or table.shape[1] <= 1:
            return table
        return table.iloc[:-1, :-1]

    def _pearson(self, x, y):
        """
        Compute Pearson correlation coefficient.

        Parameters
        ----------
        x, y : pd.Series
            Variables to correlate.

        Returns
        -------
        r : float
            Correlation coefficient.
        p_val : float
            Two-tailed p-value.
        """
        return stats.pearsonr(x, y)

    def _t_test(self, groups: list):
        """
        Perform independent samples t-test (Welch's t-test).

        Parameters
        ----------
        groups : list of pd.Series
            Two groups to compare (length must be 2).

        Returns
        -------
        statistic : float
            T-statistic.
        p_val : float
            Two-tailed p-value.

        Notes
        -----
        Uses Welch's t-test (equal_var=False) which does not assume equal variances.
        """
        return stats.ttest_ind(groups[0], groups[1], equal_var=False)

    def _anova(self, groups: list):
        """
        Perform one-way ANOVA F-test.

        Parameters
        ----------
        groups : list of pd.Series
            Groups to compare (length must be >= 2).

        Returns
        -------
        statistic : float
            F-statistic.
        p_val : float
            P-value.
        """
        return stats.f_oneway(*groups)

    def _fisher(self, table: pd.DataFrame):
        """
        Perform Fisher's exact test for 2x2 contingency tables.

        Parameters
        ----------
        table : pd.DataFrame
            2x2 contingency table.

        Returns
        -------
        statistic : float
            Odds ratio.
        p_val : float
            Two-tailed p-value.

        Raises
        ------
        ValueError
            If table is not 2x2.

        Notes
        -----
        Fisher's exact test is used when expected cell counts are <5 in a 2x2 table,
        where chi-square test assumptions are violated.
        """
        if table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires a 2x2 table")
        return stats.fisher_exact(table)

    def _chi_square(self, table: pd.DataFrame):
        """
        Perform chi-square test of independence.

        Parameters
        ----------
        table : pd.DataFrame
            Contingency table.

        Returns
        -------
        statistic : float
            Chi-square statistic.
        p_val : float
            P-value.
        dof : int
            Degrees of freedom.
        expected : ndarray
            Expected frequencies.
        """
        return stats.chi2_contingency(table)


class ElasticNetVariableSelection:
    """
    Perform variable selection using elastic net regularization with cross-validation.

    Elastic net combines L1 (Lasso) and L2 (Ridge) penalties, providing both
    variable selection and handling of multicollinearity. Variables with non-zero
    coefficients after regularization are selected.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing response and explanatory variables.
    response_variable : str
        Name of the response/outcome variable.
    explanatory_variables : list of str
        List of candidate predictor variable names.
    outcome_type : str, default 'binary'
        Type of outcome variable:
        - 'binary': Binary outcomes (uses logistic regression)
        - 'count': Count outcomes (uses Poisson-like regression)
    alpha_ratio : float, default 0.5
        L1 ratio for elastic net, must be in [0, 1]:
        - 0.0: Pure Ridge regression (L2 penalty only)
        - 0.5: Balanced elastic net (equal L1 and L2)
        - 1.0: Pure Lasso (L1 penalty only)
        Default of 0.5 balances variable selection and multicollinearity handling.
    cv : int, default 5
        Number of cross-validation folds for regularization parameter selection.

    Attributes
    ----------
    df : pd.DataFrame
        Subset DataFrame with response and explanatory variables.
    response_variable : str
        Response variable name.
    explanatory_variables : list of str
        Explanatory variable names.
    outcome_type : str
        Type of outcome ('binary' or 'count').
    alpha_ratio : float
        L1 ratio used.
    cv : int
        Number of CV folds used.
    selected_explanatory_variables : list of str
        Original variable names selected by elastic net.
    results_df : pd.DataFrame
        Results showing coefficients for all one-hot encoded features.
    model : fitted model
        Trained elastic net model (LogisticRegressionCV or ElasticNetCV).
    X_prepared : pd.DataFrame
        Preprocessed feature matrix used for fitting.
    y_prepared : pd.Series
        Preprocessed response used for fitting.

    Examples
    --------
    >>> selector = ElasticNetVariableSelection(
    ...     df=cleaned_df,
    ...     response_variable='SSI',
    ...     explanatory_variables=['Age', 'Sex', 'BMI', 'Diabetes', 'INPWT'],
    ...     outcome_type='binary',
    ...     alpha_ratio=0.5,
    ...     cv=5
    ... )
    Elastic Net selected 3 of 5 variables
    Selected variables: ['BMI', 'Diabetes', 'INPWT']

    >>> print(selector.results_df)
              Variable  Coefficient  Abs_Coefficient  Selected
    0             INPWT       0.8234           0.8234      True
    1          Diabetes       0.4521           0.4521      True
    2               BMI       0.3012           0.3012      True
    3        Sex_Female       0.0000           0.0000     False
    4               Age       0.0000           0.0000     False

    See Also
    --------
    UnivariateVariableSelection : Univariate statistical testing for selection.
    sklearn.linear_model.LogisticRegressionCV : Logistic regression with CV.
    sklearn.linear_model.ElasticNetCV : Elastic net regression with CV.

    Notes
    -----
    Advantages of elastic net for variable selection:
    - Handles multicollinearity better than univariate methods
    - Performs automatic variable selection (L1 penalty)
    - Maintains stability with correlated predictors (L2 penalty)
    - Cross-validation prevents overfitting
    - Works with p >> n (more predictors than observations)

    The alpha_ratio parameter controls the penalty balance:
    - Lower values (closer to 0) emphasize multicollinearity handling
    - Higher values (closer to 1) emphasize sparse solutions
    - Default 0.5 is generally robust for clinical data

    Preprocessing is automatic and includes:
    - Standardization of numeric variables (required for regularization)
    - One-hot encoding of categorical variables
    - Removal of rare categories (<5 observations)
    - Complete case analysis (missing data removed)

    References
    ----------
    .. [1] Zou H, Hastie T. "Regularization and variable selection via the elastic net."
           J R Stat Soc Series B Stat Methodol. 2005;67(2):301-320.
    .. [2] Friedman J, Hastie T, Tibshirani R. "Regularization paths for generalized
           linear models via coordinate descent." J Stat Softw. 2010;33(1):1-22.
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
        """
        Prepare data for elastic net using DataPreparer.

        Applies standardization and one-hot encoding required for regularization.
        """
        preparer = DataPreparer(
            df=self.df,
            response_variable=self.response_variable,
            explanatory_variables=self.explanatory_variables,
        )

        self.X_prepared, self.y_prepared = preparer.preprocess(
            drop_first_category=True,
            scale_numeric=True,  # Required for elastic net
            min_category_size=1,
            check_separation=False,  # Not needed for regularized models
        )

    def _fit_elastic_net(self):
        """
        Fit elastic net model with cross-validation for regularization parameter selection.

        Raises
        ------
        ValueError
            If outcome_type is not 'binary' or 'count'.
        """
        if self.outcome_type == "binary":
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
        """
        Extract variables with non-zero coefficients and map back to original names.

        Creates results_df with coefficient information and populates
        selected_explanatory_variables with original variable names.
        """
        # Get coefficients
        if self.outcome_type == "binary":
            coefs = self.model.coef_[0]
        else:
            coefs = self.model.coef_

        # Create results dataframe for one-hot encoded features
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
            # Check if it's a one-hot encoded variable
            found = False
            for orig_var in self.explanatory_variables:
                if encoded_var.startswith(orig_var + "_"):
                    original_vars.add(orig_var)
                    found = True
                    break

            # If not one-hot encoded, it's a numeric variable
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


class NestedCVElasticNetVariableSelection:
    """
    Nested cross-validation wrapper for elastic net variable selection with stability analysis.

    Provides honest performance estimates and selection stability by wrapping
    ElasticNetVariableSelection in an outer cross-validation loop. This addresses
    optimistic bias in single-pass selection and quantifies how reliably variables
    are selected across different data subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing response and explanatory variables.
    response_variable : str
        Name of the response/outcome variable.
    explanatory_variables : list of str
        List of candidate predictor variable names.
    outcome_type : str, default 'binary'
        Type of outcome: 'binary' or 'count'.
    alpha_ratio : float, default 0.5
        L1 ratio for elastic net (0=Ridge, 0.5=Elastic Net, 1=Lasso).
    inner_cv : int, default 5
        Number of CV folds for hyperparameter tuning (inner loop).
        Used within ElasticNetVariableSelection to tune regularization strength.
    outer_cv : int, default 5
        Number of CV folds for performance estimation (outer loop).
        Each fold provides independent test set for unbiased evaluation.
    n_repeats : int, default 1
        Number of times to repeat the outer CV with different random splits.
        n_repeats=1 gives standard k-fold CV. n_repeats=10 with outer_cv=5
        gives 50 total iterations for smoother selection frequency estimates.
        Recommended: 10 repeats for n<200 to reduce sampling variance.
    random_state : int, default 42
        Random seed for reproducibility of CV splits.
    selection_threshold : float, default 0.5
        Minimum selection frequency (0 to 1) for a variable to be included
        in consensus set. Default 0.5 means variable must be selected in
        ≥50% of outer folds. Common alternatives: 0.7 (70%), 0.8 (80%).

    Attributes
    ----------
    selected_explanatory_variables : list of str
        Consensus variables selected in ≥selection_threshold of outer folds.
        Sorted by selection frequency (descending).
    selection_frequencies : dict of {str: float}
        Selection frequency for each variable across outer folds.
        Values range from 0.0 (never selected) to 1.0 (always selected).
    coefficient_distributions : dict of {str: list of float}
        Distribution of coefficients for each variable across folds where selected.
        Quantifies coefficient stability and magnitude.
    performance_metrics : dict
        Test performance metrics aggregated across outer folds.
        For binary outcomes: auc_mean, auc_std, auc_min, auc_max, brier_mean, brier_std.
    fold_results : list of dict
        Detailed results for each outer fold including selected variables,
        performance metrics, and sample sizes.

    Examples
    --------
    >>> # Standard 5-fold CV
    >>> selector = NestedCVElasticNetSelection(
    ...     df=cleaned_df,
    ...     response_variable='SSI',
    ...     explanatory_variables=predictors,
    ...     outcome_type='binary',
    ...     alpha_ratio=0.5,
    ...     outer_cv=5,
    ...     n_repeats=1  # Single 5-fold CV
    ... )
    >>>
    >>> # Repeated 5-fold CV for smoother estimates (RECOMMENDED)
    >>> selector = NestedCVElasticNetSelection(
    ...     df=cleaned_df,
    ...     response_variable='SSI',
    ...     explanatory_variables=predictors,
    ...     outcome_type='binary',
    ...     alpha_ratio=0.5,
    ...     outer_cv=5,
    ...     n_repeats=10  # 50 total iterations
    ... )
    >>>
    >>> # Get consensus variables (selected in ≥50% of folds)
    >>> print(selector.selected_explanatory_variables)
    ['Diabetes', 'BMI', 'INPWT']
    >>>
    >>> # View detailed stability report
    >>> selector.print_stability_report()
    >>>
    >>> # Export results for publication
    >>> results_df = selector.get_results_dataframe()
    >>> results_df.to_csv('selection_stability.csv', index=False)

    See Also
    --------
    ElasticNetVariableSelection : Single-pass elastic net selection.
    UnivariateVariableSelection : Univariate statistical testing for selection.

    Notes
    -----
    The nested CV process:

    1. **Outer loop (performance estimation)**: Data split into k folds
       - Each fold: ~80% training, ~20% testing
       - Test sets are never used for variable selection or training

    2. **Inner loop (variable selection)**: For each training set
       - Run ElasticNetVariableSelection with cross-validation
       - Select variables, fit model, predict on test set
       - Record selected variables and performance

    3. **Aggregation**: Across all outer folds
       - Calculate selection frequencies
       - Identify consensus variables (≥threshold)
       - Aggregate performance metrics (mean, SD, range)
       - Report coefficient distributions

    **Advantages over single-pass selection:**

    - **Unbiased performance**: Test AUC represents expected performance on
      new patients, not optimistically biased training performance
    - **Selection stability**: Quantifies reproducibility of variable selection
      across different data subsets
    - **Coefficient uncertainty**: Distribution shows variability in effect sizes
    - **Robust inference**: Variables selected consistently are more likely to
      replicate in external validation

    **Selection threshold interpretation:**

    - **0.5 (50%)**: Liberal, includes moderately stable variables
    - **0.7 (70%)**: Moderate, requires fairly consistent selection
    - **0.8 (80%)**: Conservative, only highly stable variables
    - **1.0 (100%)**: Very conservative, only variables selected in every fold

    For small samples (n<200), lower thresholds (0.5) may be necessary to
    avoid excluding important variables due to sampling variability.

    **Computational cost:**

    With outer_cv=5 and inner_cv=5, this runs ~25x slower than single-pass
    selection (5 outer folds × ~5 inner folds each). Consider reducing outer_cv
    to 3 for faster results with reasonable stability estimates.

    References
    ----------
    .. [1] Varma S, Simon R. "Bias in error estimation when using
           cross-validation for model selection." BMC Bioinformatics. 2006;7:91.
    .. [2] Meinshausen N, Bühlmann P. "Stability selection."
           J R Stat Soc Series B Stat Methodol. 2010;72(4):417-473.
    .. [3] Heinze G, Wallisch C, Dunkler D. "Variable selection - A review and
           recommendations for the practicing statistician."
           Biom J. 2018;60(3):431-449.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        response_variable: str,
        explanatory_variables: List[str],
        outcome_type: str = "binary",
        alpha_ratio: float = 0.5,
        inner_cv: int = 5,
        outer_cv: int = 5,
        n_repeats: int = 1,
        random_state: int = 42,
        selection_threshold: float = 0.5,
    ):
        self.df = df
        self.response_variable = response_variable
        self.explanatory_variables = explanatory_variables
        self.outcome_type = outcome_type
        self.alpha_ratio = alpha_ratio
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.selection_threshold = selection_threshold

        # Validate selection threshold
        if not 0 < selection_threshold <= 1.0:
            raise ValueError("selection_threshold must be between 0 and 1")

        # Results storage
        self.selection_frequencies: Dict[str, float] = {}
        self.coefficient_distributions: Dict[str, List[float]] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.fold_results: List[Dict] = []
        self.selected_explanatory_variables: List[str] = []

        # Run nested CV
        self._run_nested_cv()
        self._aggregate_results()

    def _run_nested_cv(self):
        """
        Execute nested cross-validation with outer and inner loops.

        Outer loop: Honest performance estimation on held-out test sets.
        Inner loop: Variable selection with hyperparameter tuning on training data.
        """
        print(f"\n{'='*60}")
        print(f"Running Nested CV Elastic Net Selection")
        print(
            f"Outer CV: {self.outer_cv}-fold × {self.n_repeats} repeats = {self.outer_cv * self.n_repeats} iterations"
        )
        print(f"Inner CV: {self.inner_cv}-fold")
        print(f"Selection threshold: {self.selection_threshold*100:.0f}%")
        print(f"{'='*60}\n")

        # Prepare data for splitting
        df_subset = self.df[
            [self.response_variable] + self.explanatory_variables
        ].copy()
        df_subset = df_subset.dropna(subset=[self.response_variable])

        y_all = df_subset[self.response_variable]

        # Outer CV loop with optional repeats
        if self.n_repeats > 1:
            outer_cv_splitter = RepeatedStratifiedKFold(
                n_splits=self.outer_cv,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
        else:
            outer_cv_splitter = StratifiedKFold(
                n_splits=self.outer_cv, shuffle=True, random_state=self.random_state
            )

        total_folds = self.outer_cv * self.n_repeats

        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv_splitter.split(df_subset, y_all)
        ):
            repeat_num = fold_idx // self.outer_cv + 1
            fold_in_repeat = fold_idx % self.outer_cv + 1

            print(f"\n{'─'*60}")
            if self.n_repeats > 1:
                print(
                    f"Repeat {repeat_num}/{self.n_repeats}, Fold {fold_in_repeat}/{self.outer_cv} (Overall: {fold_idx + 1}/{total_folds})"
                )
            else:
                print(f"Fold {fold_idx + 1}/{self.outer_cv}")
            print(f"{'─'*60}")

            # Split data
            df_train = df_subset.iloc[train_idx]
            df_test = df_subset.iloc[test_idx]

            print(f"Training size: {len(df_train)}, Test size: {len(df_test)}")

            # Variable selection on training data using inner CV
            selector = ElasticNetVariableSelection(
                df=df_train,
                response_variable=self.response_variable,
                explanatory_variables=self.explanatory_variables,
                outcome_type=self.outcome_type,
                alpha_ratio=self.alpha_ratio,
                cv=self.inner_cv,
            )

            selected_vars = selector.selected_explanatory_variables
            print(f"Selected variables ({len(selected_vars)}): {selected_vars}")

            if len(selected_vars) == 0:
                print("WARNING: No variables selected in this fold!")
                continue

            # Prepare training data with selected variables
            preparer_train = DataPreparer(
                df=df_train,
                response_variable=self.response_variable,
                explanatory_variables=selected_vars,
            )

            try:
                X_train, y_train = preparer_train.preprocess(
                    drop_first_category=True,
                    scale_numeric=False,
                    check_separation=False,
                )
            except ValueError as e:
                print(f"WARNING: Training set preprocessing failed: {e}")
                print("Skipping this fold")
                continue

            # Check if training set is too small
            if len(X_train) < 10:
                print(
                    f"WARNING: Training set too small after preprocessing (n={len(X_train)}). Skipping this fold."
                )
                continue

            # Fit model
            if self.outcome_type == "binary":
                from .modeling import LogisticRegressionModel

                model = LogisticRegressionModel()
            else:
                from .modeling import NegativeBinomialRegressionModel

                model = NegativeBinomialRegressionModel()

            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"WARNING: Model fitting failed: {e}")
                print(
                    "Skipping this fold (likely due to perfect separation or multicollinearity)"
                )
                continue

            # Check convergence
            if not model.check_convergence():
                print("WARNING: Model did not converge. Skipping this fold.")
                continue

            # Store coefficients
            for var in selected_vars:
                if var in X_train.columns:
                    coef = model.model.params[var]
                    if var not in self.coefficient_distributions:
                        self.coefficient_distributions[var] = []
                    self.coefficient_distributions[var].append(coef)

            # Prepare test data
            preparer_test = DataPreparer(
                df=df_test,
                response_variable=self.response_variable,
                explanatory_variables=selected_vars,
            )

            try:
                X_test, y_test = preparer_test.preprocess(
                    drop_first_category=True,
                    scale_numeric=False,
                    check_separation=False,
                )
            except ValueError as e:
                print(f"WARNING: Test set preprocessing failed: {e}")
                print(
                    "Skipping this fold (likely due to rare categories or missing data)"
                )
                continue

            # Check if test set is empty after preprocessing
            if len(X_test) == 0 or len(y_test) == 0:
                print(
                    "WARNING: Test set empty after preprocessing. Skipping this fold."
                )
                continue

            # Predict on test set
            X_test_const = sm.add_constant(X_test, has_constant="add")
            y_pred_proba = model.model.predict(X_test_const)

            # Calculate performance metrics
            if self.outcome_type == "binary":
                auc = roc_auc_score(y_test, y_pred_proba)
                brier = brier_score_loss(y_test, y_pred_proba)
                print(f"Test AUC: {auc:.3f}, Brier Score: {brier:.3f}")

                fold_result = {
                    "fold": fold_idx + 1,
                    "n_train": len(df_train),
                    "n_test": len(df_test),
                    "n_selected": len(selected_vars),
                    "selected_vars": selected_vars,
                    "auc": auc,
                    "brier": brier,
                }
            else:
                # For count outcomes, could add other metrics
                fold_result = {
                    "fold": fold_idx + 1,
                    "n_train": len(df_train),
                    "n_test": len(df_test),
                    "n_selected": len(selected_vars),
                    "selected_vars": selected_vars,
                }

            self.fold_results.append(fold_result)

    def _aggregate_results(self):
        """
        Aggregate selection frequencies, coefficients, and performance across folds.

        Identifies consensus variables based on selection_threshold and computes
        summary statistics for reporting.
        """
        print(f"\n{'='*60}")
        print("Aggregating Results Across Folds")
        print(f"{'='*60}\n")

        if len(self.fold_results) == 0:
            print("ERROR: No folds completed successfully!")
            print("Cannot perform variable selection - all folds failed.")
            self.selected_explanatory_variables = []
            return

        print(f"Successfully completed {len(self.fold_results)} folds")

        # Calculate selection frequencies
        selection_counts = {}
        for fold_result in self.fold_results:
            for var in fold_result["selected_vars"]:
                selection_counts[var] = selection_counts.get(var, 0) + 1

        for var, count in selection_counts.items():
            self.selection_frequencies[var] = count / len(self.fold_results)

        # Identify consensus variables (selected in ≥threshold of folds)
        self.selected_explanatory_variables = [
            var
            for var, freq in self.selection_frequencies.items()
            if freq >= self.selection_threshold
        ]

        # Sort by selection frequency (descending)
        self.selected_explanatory_variables.sort(
            key=lambda v: self.selection_frequencies[v], reverse=True
        )

        # Aggregate performance metrics
        if self.outcome_type == "binary" and len(self.fold_results) > 0:
            aucs = [r["auc"] for r in self.fold_results if "auc" in r]
            briers = [r["brier"] for r in self.fold_results if "brier" in r]

            if aucs:
                self.performance_metrics = {
                    "auc_mean": np.mean(aucs),
                    "auc_std": np.std(aucs),
                    "auc_min": np.min(aucs),
                    "auc_max": np.max(aucs),
                    "brier_mean": np.mean(briers),
                    "brier_std": np.std(briers),
                }

                print(f"Test Performance (n={len(aucs)} folds):")
                print(
                    f"  AUC: {self.performance_metrics['auc_mean']:.3f} ± "
                    f"{self.performance_metrics['auc_std']:.3f} "
                    f"[{self.performance_metrics['auc_min']:.3f}, "
                    f"{self.performance_metrics['auc_max']:.3f}]"
                )
                print(
                    f"  Brier: {self.performance_metrics['brier_mean']:.3f} ± "
                    f"{self.performance_metrics['brier_std']:.3f}"
                )

        print(
            f"\nConsensus Variables (selected in ≥{self.selection_threshold*100:.0f}% of folds):"
        )
        print(
            f"  {len(self.selected_explanatory_variables)} variables: "
            f"{self.selected_explanatory_variables}"
        )

    def print_stability_report(self):
        """
        Print comprehensive stability and performance report.

        Displays selection frequencies, coefficient distributions, and test
        performance metrics in a formatted table for easy interpretation.

        Examples
        --------
        >>> selector.print_stability_report()
        ======================================================================
        VARIABLE SELECTION STABILITY REPORT
        ======================================================================
        ...
        """
        print(f"\n{'='*70}")
        print("VARIABLE SELECTION STABILITY REPORT")
        print(f"{'='*70}\n")

        print(
            f"Analysis: {self.response_variable} ~ {len(self.explanatory_variables)} predictors"
        )
        print(f"Method: Nested CV Elastic Net (alpha_ratio={self.alpha_ratio})")
        if self.n_repeats > 1:
            print(
                f"Outer CV: {self.outer_cv}-fold × {self.n_repeats} repeats = {self.outer_cv * self.n_repeats} total iterations"
            )
        else:
            print(f"Outer CV: {self.outer_cv}-fold")
        print(f"Inner CV: {self.inner_cv}-fold\n")

        # Selection frequency table
        print("Variable Selection Frequency:")
        print(f"{'─'*70}")
        print(f"{'Variable':<30} {'Frequency':<15} {'Coefficient (mean±SD)':<25}")
        print(f"{'─'*70}")

        # Sort by frequency (descending)
        sorted_vars = sorted(
            self.selection_frequencies.items(), key=lambda x: (-x[1], x[0])
        )

        for var, freq in sorted_vars:
            freq_str = f"{freq*100:.0f}% ({int(freq*len(self.fold_results))}/{len(self.fold_results)})"

            if var in self.coefficient_distributions:
                coefs = self.coefficient_distributions[var]
                coef_mean = np.mean(coefs)
                coef_std = np.std(coefs)
                coef_str = f"{coef_mean:+.3f} ± {coef_std:.3f}"
            else:
                coef_str = "N/A"

            marker = "✓" if freq >= self.selection_threshold else " "
            print(f"{marker} {var:<28} {freq_str:<15} {coef_str:<25}")

        print(f"{'─'*70}\n")

        # Performance metrics
        if self.performance_metrics:
            print("Test Set Performance:")
            print(f"{'─'*70}")
            print(
                f"  AUC:         {self.performance_metrics['auc_mean']:.3f} ± "
                f"{self.performance_metrics['auc_std']:.3f}"
            )
            print(
                f"  AUC range:   [{self.performance_metrics['auc_min']:.3f}, "
                f"{self.performance_metrics['auc_max']:.3f}]"
            )
            print(
                f"  Brier score: {self.performance_metrics['brier_mean']:.3f} ± "
                f"{self.performance_metrics['brier_std']:.3f}"
            )
            print(f"{'─'*70}\n")

        # Consensus variables
        print(f"Consensus Variables (≥{self.selection_threshold*100:.0f}% selection):")
        print(f"{'─'*70}")
        if self.selected_explanatory_variables:
            for var in self.selected_explanatory_variables:
                freq = self.selection_frequencies[var]
                print(f"  • {var}: {freq*100:.0f}% of folds")
        else:
            print("  None (no variables met threshold)")
        print(f"{'─'*70}\n")

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Return stability results as pandas DataFrame for export or further analysis.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - variable: str, variable name
            - selection_frequency: float, proportion of folds (0-1)
            - n_folds_selected: int, number of folds selected
            - coef_mean: float, mean coefficient across folds
            - coef_std: float, standard deviation of coefficient
            - consensus_selected: bool, whether in consensus set

        Examples
        --------
        >>> results_df = selector.get_results_dataframe()
        >>> results_df.to_csv('selection_stability.csv', index=False)
        >>>
        >>> # Filter to consensus variables
        >>> consensus = results_df[results_df['consensus_selected']]
        """
        results = []
        for var in self.explanatory_variables:
            freq = self.selection_frequencies.get(var, 0.0)

            if var in self.coefficient_distributions:
                coefs = self.coefficient_distributions[var]
                coef_mean = np.mean(coefs)
                coef_std = np.std(coefs)
            else:
                coef_mean = np.nan
                coef_std = np.nan

            results.append(
                {
                    "variable": var,
                    "selection_frequency": freq,
                    "n_folds_selected": int(freq * len(self.fold_results)),
                    "coef_mean": coef_mean,
                    "coef_std": coef_std,
                    "consensus_selected": freq >= self.selection_threshold,
                }
            )

        df = pd.DataFrame(results)
        df = df.sort_values("selection_frequency", ascending=False)
        return df
