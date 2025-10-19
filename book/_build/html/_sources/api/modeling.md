# Statistical Modeling

The `modeling` module provides regression models for binary and count outcomes with automatic convergence checking.

## BaseModel

```{eval-rst}
.. autoclass:: statwise.modeling.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## LogisticRegressionModel

```{eval-rst}
.. autoclass:: statwise.modeling.LogisticRegressionModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## NegativeBinomialRegressionModel

```{eval-rst}
.. autoclass:: statwise.modeling.NegativeBinomialRegressionModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Logistic Regression for Binary Outcomes

```python
from statwise.modeling import LogisticRegressionModel
import numpy as np

# Fit model
model = LogisticRegressionModel()
fitted = model.fit(X, y)

# Check convergence
if not model.check_convergence():
    print("WARNING: Model did not converge - check for separation")

# View full summary
print(model.summary())

# Extract key results
coefficients = fitted.params
std_errors = fitted.bse
p_values = fitted.pvalues
conf_int = fitted.conf_int()

# Calculate odds ratios and 95% CIs
odds_ratios = np.exp(coefficients)
or_ci_lower = np.exp(conf_int[0])
or_ci_upper = np.exp(conf_int[1])

print("\nOdds Ratios (95% CI):")
for var in X.columns:
    or_val = odds_ratios[var]
    ci_low = or_ci_lower[var]
    ci_high = or_ci_upper[var]
    p_val = p_values[var]
    print(f"{var}: OR={or_val:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f}), p={p_val:.3f}")
```

Example output:
```
Odds Ratios (95% CI):
const: OR=0.15 (95% CI: 0.08-0.28), p<0.001
Age: OR=1.02 (95% CI: 0.99-1.05), p=0.234
BMI: OR=1.08 (95% CI: 1.01-1.16), p=0.032
Sex_Male: OR=1.45 (95% CI: 0.78-2.70), p=0.242
Diabetes_Yes: OR=2.34 (95% CI: 1.21-4.52), p=0.012
INPWT_Yes: OR=0.62 (95% CI: 0.31-1.24), p=0.175
```

### Negative Binomial for Count Outcomes

```python
from statwise.modeling import NegativeBinomialRegressionModel
import numpy as np

# Fit model
model = NegativeBinomialRegressionModel()
fitted = model.fit(X, y_los)  # Length of stay

# Check convergence
model.check_convergence()

# View summary
print(model.summary())

# Calculate incidence rate ratios
irr = np.exp(fitted.params)
irr_ci_lower = np.exp(fitted.conf_int()[0])
irr_ci_upper = np.exp(fitted.conf_int()[1])

print("\nIncidence Rate Ratios (95% CI):")
for var in X.columns:
    irr_val = irr[var]
    ci_low = irr_ci_lower[var]
    ci_high = irr_ci_upper[var]
    p_val = fitted.pvalues[var]
    print(f"{var}: IRR={irr_val:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f}), p={p_val:.3f}")

# Check dispersion
alpha = fitted.params['alpha']
print(f"\nDispersion parameter (alpha): {alpha:.3f}")
if alpha > 0.5:
    print("Substantial overdispersion detected - NB model appropriate")
```

### Comparing Multiple Models

```python
from statwise.modeling import LogisticRegressionModel

# Unadjusted (treatment only)
model_unadj = LogisticRegressionModel()
model_unadj.fit(X[['INPWT_Yes']], y)

# Univariate-adjusted
model_univ = LogisticRegressionModel()
model_univ.fit(X[univariate_selected], y)

# Elastic net-adjusted
model_en = LogisticRegressionModel()
model_en.fit(X[elasticnet_selected], y)

# Consensus-adjusted
model_consensus = LogisticRegressionModel()
model_consensus.fit(X[consensus_selected], y)

# Compare treatment effect across models
models = {
    'Unadjusted': model_unadj,
    'Univariate-adjusted': model_univ,
    'Elastic net-adjusted': model_en,
    'Consensus-adjusted': model_consensus
}

print("Treatment Effect (INPWT) Across Models:")
print("-" * 60)
for name, model in models.items():
    coef = model.model.params['INPWT_Yes']
    se = model.model.bse['INPWT_Yes']
    or_val = np.exp(coef)
    ci = np.exp(model.model.conf_int().loc['INPWT_Yes'])
    p_val = model.model.pvalues['INPWT_Yes']
    
    print(f"{name}:")
    print(f"  OR = {or_val:.2f} (95% CI: {ci[0]:.2f}-{ci[1]:.2f}), p={p_val:.3f}")
```

## Interpreting Results

### Logistic Regression

**Odds Ratio (OR) = exp(coefficient)**

- **OR = 1.0:** No effect (predictor doesn't change odds)
- **OR > 1.0:** Increases odds
  - OR = 1.5: 50% increase in odds
  - OR = 2.0: 100% increase (doubling) in odds
  - OR = 3.0: 200% increase (tripling) in odds
- **OR < 1.0:** Decreases odds
  - OR = 0.5: 50% decrease (halving) in odds
  - OR = 0.33: 67% decrease in odds

**Example interpretations:**
- `Diabetes_Yes: OR=2.34` → Patients with diabetes have 2.34 times the odds of SSI compared to non-diabetics
- `INPWT_Yes: OR=0.62` → Patients receiving INPWT have 38% lower odds of SSI compared to standard care

**Note:** Odds ratios approximate risk ratios when outcomes are rare (<10% prevalence).

### Negative Binomial Regression

**Incidence Rate Ratio (IRR) = exp(coefficient)**

- **IRR = 1.0:** No effect on expected count
- **IRR > 1.0:** Increases expected count
  - IRR = 1.2: 20% increase in expected count
  - IRR = 1.5: 50% increase in expected count
- **IRR < 1.0:** Decreases expected count
  - IRR = 0.8: 20% decrease in expected count
  - IRR = 0.5: 50% decrease in expected count

**Example interpretations:**
- `INPWT_Yes: IRR=0.85` → INPWT reduces expected length of stay by 15%
- `Diabetes_Yes: IRR=1.35` → Diabetes increases expected length of stay by 35%

### P-values and Confidence Intervals

**P-value interpretation:**
- p < 0.05: Statistically significant at conventional level
- p < 0.01: Highly significant
- p < 0.001: Very highly significant

**Confidence interval interpretation:**
- If 95% CI excludes 1.0 (OR or IRR): Significant at p<0.05
- If 95% CI includes 1.0: Not significant at p<0.05
- Wider CIs indicate greater uncertainty (smaller sample, rare events)

### Model Fit Statistics

**AIC (Akaike Information Criterion):**
- Lower is better
- Use to compare non-nested models
- Penalizes model complexity

**BIC (Bayesian Information Criterion):**
- Lower is better
- More strongly penalizes complexity than AIC
- Useful for model selection

**Pseudo-R²:**
- Ranges from 0 to 1
- Not directly comparable to R² in linear regression
- Values >0.2 indicate reasonably good fit

**Log-Likelihood:**
- Higher (less negative) is better
- Used in likelihood ratio tests

## Convergence Issues

### Causes

1. **Separation** - Perfect or quasi-perfect prediction
2. **Multicollinearity** - Highly correlated predictors
3. **Small sample size** - Too few events per variable
4. **Numerical instability** - Extreme parameter values

### Diagnostics

```python
# Check convergence
if not model.check_convergence():
    # Inspect coefficients and standard errors
    results = model.model.params
    std_errors = model.model.bse
    
    # Look for warning signs:
    # - Very large coefficients (>10)
    # - Very large standard errors (>10)
    # - Coefficients approaching infinity
    
    problematic = results[np.abs(results) > 10]
    if len(problematic) > 0:
        print(f"Large coefficients detected: {problematic.index.tolist()}")
```

### Solutions

1. **Check for separation:**
```python
from statwise.preparation import DataPreparer
preparer = DataPreparer(df, outcome, predictors)
problematic = preparer.check_separation(verbose=True)
```

2. **Use penalized regression:**
```python
from statsmodels.discrete.discrete_model import Logit

# Firth's penalized likelihood
X_const = sm.add_constant(X)
firth_model = Logit(y, X_const).fit_regularized(method='l1', alpha=0.01)
```

3. **Reduce predictors:**
```python
# Use elastic net for variable selection
from statwise.selection import ElasticNetVariableSelection
selector = ElasticNetVariableSelection(df, outcome, predictors)
reduced_predictors = selector.selected_explanatory_variables
```

4. **Increase sample size** or **merge categories** to reduce sparsity

## Model Assumptions

### Logistic Regression

1. **Binary outcome** - Response must be 0/1
2. **Independence** - Observations are independent
3. **Linearity** - Log odds linear in predictors (for continuous predictors)
4. **No perfect multicollinearity** - Predictors not perfectly correlated
5. **Large sample** - Sufficient events per variable (~10 EPV rule)

**Check linearity for continuous predictors:**
```python
# Add quadratic term if non-linear
X['BMI_squared'] = X['BMI'] ** 2
```

### Negative Binomial Regression

1. **Non-negative integer outcome** - Counts (0, 1, 2, ...)
2. **Independence** - Observations are independent
3. **Overdispersion present** - Variance > mean
4. **Log-linear relationship** - Log of expected count linear in predictors

**Check for overdispersion:**
```python
# Fit Poisson first
from statsmodels.discrete.discrete_model import Poisson
X_const = sm.add_constant(X)
poisson_model = Poisson(y, X_const).fit()

# Check deviance
residual_deviance = poisson_model.deviance
df = poisson_model.df_resid

if residual_deviance / df > 1.5:
    print("Overdispersion detected - use Negative Binomial")
else:
    print("Poisson may be sufficient")
```

## Complete Workflow Example

```python
from statwise.preparation import DataPreparer
from statwise.modeling import LogisticRegressionModel
import numpy as np
import pandas as pd

# Prepare data
preparer = DataPreparer(df, 'SSI', selected_predictors)
X, y = preparer.preprocess(
    drop_first_category=True,
    scale_numeric=False,
    check_separation=True
)

# Fit model
model = LogisticRegressionModel()
fitted = model.fit(X, y)

# Check convergence
if not model.check_convergence():
    print("WARNING: Convergence issues detected")
    
# Display summary
print(model.summary())

# Extract and format results
results_df = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': fitted.params[X.columns],
    'Std Error': fitted.bse[X.columns],
    'OR': np.exp(fitted.params[X.columns]),
    'OR 95% CI Lower': np.exp(fitted.conf_int()[0][X.columns]),
    'OR 95% CI Upper': np.exp(fitted.conf_int()[1][X.columns]),
    'P-value': fitted.pvalues[X.columns]
})

results_df['Significant'] = results_df['P-value'] < 0.05
print("\n", results_df.to_string(index=False))

# Save results
results_df.to_csv('logistic_regression_results.csv', index=False)
```