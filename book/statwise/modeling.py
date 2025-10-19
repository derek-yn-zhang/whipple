import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.discrete.discrete_model import NegativeBinomial
from typing import Optional


class BaseModel:
    """
    Base class for regression models providing common interface and convergence checking.

    This abstract base class defines the interface for all regression models
    and provides shared functionality for model fitting and convergence diagnostics.

    Attributes
    ----------
    model : fitted model or None
        Fitted statsmodels regression model (None until fit() is called).
    X : pd.DataFrame or None
        Feature matrix used for fitting (None until fit() is called).
    y : pd.Series or None
        Response variable used for fitting (None until fit() is called).
    converged : bool or None
        Whether maximum likelihood optimization converged successfully.
        None until convergence is checked.

    Notes
    -----
    This class should not be instantiated directly. Use subclasses:
    - LogisticRegressionModel for binary outcomes
    - NegativeBinomialRegressionModel for count outcomes
    """

    def __init__(self):
        self.model = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the regression model.

        Must be implemented by subclasses.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features).
        y : pd.Series
            Response variable with shape (n_samples,).

        Raises
        ------
        NotImplementedError
            If called on BaseModel directly rather than a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def summary(self):
        """
        Display comprehensive model summary with coefficients and fit statistics.

        Returns
        -------
        statsmodels.iolib.summary.Summary
            Summary object containing coefficient estimates, standard errors,
            z-statistics, p-values, confidence intervals, and model fit statistics.

        Raises
        ------
        ValueError
            If model has not been fitted yet.

        Examples
        --------
        >>> model = LogisticRegressionModel()
        >>> model.fit(X, y)
        >>> print(model.summary())
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        return self.model.summary()

    def check_convergence(self):
        """
        Check if maximum likelihood optimization converged successfully.

        Prints a warning message if convergence failed. Non-convergence indicates
        that parameter estimates may be unreliable due to numerical issues, perfect
        separation, or other optimization problems.

        Returns
        -------
        bool
            True if model converged or convergence status unavailable.
            False if convergence explicitly failed.

        Raises
        ------
        ValueError
            If model has not been fitted yet.

        Examples
        --------
        >>> model.fit(X, y)
        >>> if not model.check_convergence():
        ...     print("Consider using regularization or checking for separation")
        WARNING: Maximum Likelihood optimization failed to converge.
            Results may be unreliable. Check coefficients and standard errors.

        Notes
        -----
        Common causes of convergence failure:
        - Perfect or quasi-perfect separation in logistic regression
        - Multicollinearity among predictors
        - Insufficient sample size relative to number of parameters
        - Numerical instability in parameter space

        Solutions for convergence issues:
        - Use penalized regression (Firth, Ridge, elastic net)
        - Remove perfectly predictive variables
        - Increase sample size or reduce number of predictors
        - Check for separation using DataPreparer.check_separation()
        """
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
    """
    Logistic regression model for binary outcome variables.

    Fits a maximum likelihood logistic regression using statsmodels,
    automatically handling categorical responses and adding an intercept term.

    Attributes
    ----------
    model : statsmodels.discrete.discrete_model.LogitResults or None
        Fitted logistic regression model.
    X : pd.DataFrame or None
        Feature matrix used for fitting.
    y : pd.Series or None
        Binary response variable used for fitting.
    converged : bool or None
        Whether optimization converged.

    Examples
    --------
    >>> model = LogisticRegressionModel()
    >>> model.fit(X_train, y_train)
    >>> print(model.summary())
    >>>
    >>> # Check convergence
    >>> model.check_convergence()
    >>>
    >>> # Get odds ratios
    >>> odds_ratios = np.exp(model.model.params)
    >>> print(odds_ratios)

    See Also
    --------
    NegativeBinomialRegressionModel : For count outcomes.
    statsmodels.discrete.discrete_model.Logit : Underlying statsmodels class.

    Notes
    -----
    The model estimates the probability of the outcome as:

    P(Y=1|X) = 1 / (1 + exp(-(beta_0 + beta_1*X_1 + ... + beta_p*X_p)))

    Coefficients represent log odds ratios:
    - Positive coefficient: Predictor increases odds of outcome
    - Negative coefficient: Predictor decreases odds of outcome
    - exp(coefficient): Multiplicative change in odds for 1-unit increase

    For clinical interpretation:
    - Odds ratio = 1: No association
    - Odds ratio > 1: Increased odds (e.g., OR=2.0 means 2x higher odds)
    - Odds ratio < 1: Decreased odds (e.g., OR=0.5 means 50% lower odds)

    Model fitting uses maximum likelihood estimation with Newton-Raphson
    optimization. If convergence fails, consider checking for separation
    or using penalized methods.

    References
    ----------
    .. [1] Hosmer DW, Lemeshow S, Sturdivant RX. "Applied Logistic Regression."
           3rd ed. Wiley; 2013.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit logistic regression model to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features).
            Should contain preprocessed numeric and one-hot encoded variables.
        y : pd.Series
            Binary response variable with shape (n_samples,).
            Automatically converted to 0/1 integers if categorical.

        Returns
        -------
        model : statsmodels.discrete.discrete_model.LogitResults
            Fitted logistic regression model.

        Examples
        --------
        >>> from statwise.preparation import DataPreparer
        >>> preparer = DataPreparer(df, 'SSI', ['Age', 'Sex', 'BMI', 'INPWT'])
        >>> X, y = preparer.preprocess()
        >>>
        >>> model = LogisticRegressionModel()
        >>> fitted = model.fit(X, y)
        >>> print(fitted.summary())

        Notes
        -----
        Data preprocessing steps:
        1. Creates copies of X and y to avoid modifying originals
        2. Ensures X is float type for optimization
        3. Converts categorical y to 0/1 integers
        4. Adds constant term for intercept
        5. Fits using maximum likelihood
        6. Checks convergence and warns if failed

        The intercept represents the log odds when all predictors are zero
        (or at reference categories for one-hot encoded variables).
        """
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
    Negative binomial regression model for overdispersed count outcome variables.

    Fits a maximum likelihood negative binomial regression, appropriate for
    count outcomes with variance exceeding the mean (overdispersion). More
    flexible than Poisson regression for clinical count data.

    Attributes
    ----------
    model : statsmodels.discrete.discrete_model.NegativeBinomialResults or None
        Fitted negative binomial regression model.
    X : pd.DataFrame or None
        Feature matrix used for fitting.
    y : pd.Series or None
        Count response variable used for fitting.
    converged : bool or None
        Whether optimization converged.

    Examples
    --------
    >>> model = NegativeBinomialRegressionModel()
    >>> model.fit(X_train, y_los)
    >>> print(model.summary())
    >>>
    >>> # Get incidence rate ratios
    >>> irr = np.exp(model.model.params)
    >>> print(irr)

    See Also
    --------
    LogisticRegressionModel : For binary outcomes.
    statsmodels.discrete.discrete_model.NegativeBinomial : Underlying statsmodels class.

    Notes
    -----
    The negative binomial model is appropriate when:
    - Outcome is a non-negative count (0, 1, 2, ...)
    - Variance exceeds mean (overdispersion)
    - Poisson regression fits poorly (deviance >> degrees of freedom)

    Common clinical applications:
    - Length of hospital stay (days)
    - Number of complications
    - Number of readmissions
    - Healthcare utilization counts

    The model estimates expected count as:

    E[Y|X] = exp(beta_0 + beta_1*X_1 + ... + beta_p*X_p)

    Coefficients represent log incidence rate ratios:
    - Positive coefficient: Predictor increases expected count
    - Negative coefficient: Predictor decreases expected count
    - exp(coefficient): Multiplicative change in expected count

    The dispersion parameter alpha quantifies overdispersion:
    - alpha = 0: Reduces to Poisson (variance = mean)
    - alpha > 0: Overdispersion present (variance = mean + alpha*mean^2)
    - Larger alpha: Greater overdispersion

    References
    ----------
    .. [1] Hilbe JM. "Negative Binomial Regression." 2nd ed.
           Cambridge University Press; 2011.
    .. [2] Cameron AC, Trivedi PK. "Regression Analysis of Count Data."
           2nd ed. Cambridge University Press; 2013.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit negative binomial regression model to count data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features).
            Should contain preprocessed numeric and one-hot encoded variables.
        y : pd.Series
            Count response variable with shape (n_samples,).
            Must contain non-negative integers.

        Returns
        -------
        model : statsmodels.discrete.discrete_model.NegativeBinomialResults
            Fitted negative binomial regression model.

        Examples
        --------
        >>> from statwise.preparation import DataPreparer
        >>> preparer = DataPreparer(df, 'Length_of_Stay', ['Age', 'Sex', 'BMI'])
        >>> X, y = preparer.preprocess()
        >>>
        >>> model = NegativeBinomialRegressionModel()
        >>> fitted = model.fit(X, y)
        >>> print(fitted.summary())
        >>>
        >>> # Check dispersion parameter
        >>> print(f"Dispersion alpha: {fitted.params['alpha']:.3f}")

        Notes
        -----
        Data preprocessing steps:
        1. Stores X and y for reference
        2. Adds constant term for intercept
        3. Fits using maximum likelihood
        4. Checks convergence and warns if failed

        The model simultaneously estimates:
        - Beta coefficients (predictor effects on log-scale)
        - Alpha parameter (overdispersion)

        Interpretation of incidence rate ratios (IRR = exp(beta)):
        - IRR = 1: No effect on expected count
        - IRR = 1.5: 50% increase in expected count
        - IRR = 0.7: 30% decrease in expected count

        For length of stay analysis:
        - IRR = 1.2 for treatment means 20% longer expected LOS
        - IRR = 0.8 for treatment means 20% shorter expected LOS
        """
        self.X, self.y = X, y

        # Add constant term for intercept
        X_sm = sm.add_constant(self.X, has_constant="add")

        # Fit negative binomial regression
        self.model = NegativeBinomial(self.y, X_sm).fit(disp=False)

        # Check convergence
        self.check_convergence()

        return self.model
