# Data Preparation

The `preparation` module transforms cleaned data into model-ready format with standardization, encoding, and separation checking.

## DataPreparer

```{eval-rst}
.. autoclass:: statwise.preparation.DataPreparer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Basic Preprocessing

```python
from statwise.preparation import DataPreparer

preparer = DataPreparer(
    df=cleaned_df,
    response_variable='SSI',
    explanatory_variables=['Age', 'Sex', 'BMI', 'Diabetes', 'INPWT']
)

X, y = preparer.preprocess(
    drop_first_category=True,  # Create reference category
    scale_numeric=True,        # Standardize continuous variables
    min_category_size=5,       # Drop rare categories
    check_separation=True      # Warn about convergence risks
)

print(f"Final dataset: {X.shape[0]} observations, {X.shape[1]} features")
```

### Separation Checking

```python
# Check for perfect or quasi-perfect separation before modeling
preparer = DataPreparer(df, 'SSI', predictors)
problematic_vars = preparer.check_separation(threshold=0.95, verbose=True)

if problematic_vars:
    print(f"Variables with separation issues: {problematic_vars}")
    print("Consider using penalized regression or excluding these variables")
```

Example output:
```
Checking for separation issues...
Perfect separation in 'Rare_Complication': level 'Yes' has 100% in one outcome
Quasi-separation in 'High_Risk_Procedure': level 'Yes' has 96.7% in 'SSI'

WARNING: 2 predictors may cause convergence issues in standard logistic regression.
    Consider using penalized regression (Firth, Ridge) or manually excluding these variables.
```

### Different Preprocessing for Different Models

```python
# For standard logistic regression (needs separation check)
X_logistic, y = preparer.preprocess(
    drop_first_category=True,
    scale_numeric=False,  # Not required for logistic
    check_separation=True
)

# For elastic net (needs scaling, no separation check needed)
X_elasticnet, y = preparer.preprocess(
    drop_first_category=True,
    scale_numeric=True,  # Required for regularization
    check_separation=False  # Regularization handles separation
)

# For tree-based methods (minimal preprocessing)
X_tree, y = preparer.preprocess(
    drop_first_category=False,  # Trees handle multicollinearity
    scale_numeric=False,  # Trees are scale-invariant
    check_separation=False
)
```

## Understanding Separation

### What is Separation?

**Perfect separation** occurs when a predictor (or combination) perfectly predicts the outcome:
- All patients with rare complication died (100% mortality)
- No patients without treatment had the outcome (0% event rate)

**Quasi-separation** occurs when prediction is nearly perfect:
- 48 of 50 patients (96%) with high-risk procedure had SSI
- 1 of 100 patients (1%) in control group had sepsis

### Why is Separation Problematic?

Separation causes:
- Maximum likelihood estimates approaching infinity
- Standard errors approaching infinity
- Optimization algorithms failing to converge
- Unreliable p-values and confidence intervals

### Solutions for Separation

1. **Penalized regression** (Firth logistic regression, Ridge, Lasso)
2. **Exact logistic regression** (for small samples)
3. **Bayesian methods** with informative priors
4. **Variable exclusion** if not theoretically critical
5. **Category merging** to reduce separation

## Preprocessing Operations

### Standardization (Z-score)

Transforms continuous variables to mean=0, std=1:
```python
X_scaled = (X - mean(X)) / std(X)
```

**When to use:**
- Elastic net, Ridge, Lasso (required)
- Neural networks (required)
- K-nearest neighbors (recommended)

**When not to use:**
- Logistic/linear regression without penalty (optional)
- Tree-based methods (not needed)

### One-Hot Encoding

Converts categorical variables to binary indicators:

```
Sex = ['Male', 'Female', 'Male']
↓
Sex_Male = [1, 0, 1]
Sex_Female = [0, 1, 0]  # Dropped if drop_first_category=True
```

**drop_first_category=True:**
- Creates reference category
- Avoids perfect multicollinearity
- Standard for regression models

**drop_first_category=False:**
- Keeps all categories
- For models handling multicollinearity (trees, elastic net)

### Rare Category Removal

Categories with <5 observations (default) are removed to prevent:
- Unreliable coefficient estimates
- Numerical instability
- Overfitting to rare events

## Complete Case Analysis

The preparer uses complete case analysis (listwise deletion):
- Rows with any missing values in X or y are removed
- Simple and transparent
- May lose power if missingness is substantial

**Alternative:** Perform multiple imputation before calling DataPreparer.