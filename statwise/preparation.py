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

        # Convert nullable Int64 to regular int64 or float64
        for col in X.columns:
            if X[col].dtype == 'Int64':
                X[col] = X[col].astype('float64')  # Use float64 to preserve NaN

        # Convert y if it's Int64
        if y.dtype == 'Int64':
            y = y.astype('float64')

        self.X = X
        self.y = y
        return self.X, self.y
