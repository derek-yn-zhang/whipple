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
