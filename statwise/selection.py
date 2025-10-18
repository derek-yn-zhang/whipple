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
