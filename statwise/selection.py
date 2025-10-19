import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Union, Optional
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from pandas.api.types import CategoricalDtype, is_object_dtype, is_numeric_dtype

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
