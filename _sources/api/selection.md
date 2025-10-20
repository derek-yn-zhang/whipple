# Variable Selection

The `selection` module provides two complementary approaches to variable selection: univariate statistical testing and elastic net regularization.

## UnivariateVariableSelection

```{eval-rst}
.. autoclass:: statwise.selection.UnivariateVariableSelection
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## ElasticNetVariableSelection

```{eval-rst}
.. autoclass:: statwise.selection.ElasticNetVariableSelection
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## NestedCVElasticNetVariableSelection

```{eval-rst}
.. autoclass:: statwise.selection.NestedCVElasticNetVariableSelection
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Univariate Selection

```python
from statwise.selection import UnivariateVariableSelection

selector = UnivariateVariableSelection(
    df=cleaned_df,
    response_variable='SSI',
    explanatory_variables=['Age', 'Sex', 'BMI', 'Diabetes', 'ASA_Class', 'INPWT'],
    alpha=0.1  # Liberal threshold for initial screening
)

print(f"Selected {len(selector.selected_explanatory_variables)} variables")
print(selector.results_df)
```

Example output:
```
Selected 4 of 6 variables (p < 0.1)

      variable       test   p_value  statistic
0        INPWT  chi-square   0.023       5.17
1     Diabetes  chi-square   0.067       3.36
2    ASA_Class      ANOVA   0.082       2.51
3          BMI    pearson   0.089       0.28
4          Age    pearson   0.234       0.18
5          Sex  chi-square   0.445       0.58
```

### Elastic Net Selection

```python
from statwise.selection import ElasticNetVariableSelection

# For binary outcomes
selector = ElasticNetVariableSelection(
    df=cleaned_df,
    response_variable='SSI',
    explanatory_variables=predictors,
    outcome_type='binary',
    alpha_ratio=0.5,  # Balanced L1/L2 penalty
    cv=5
)

print(f"Selected variables: {selector.selected_explanatory_variables}")
print(selector.results_df.head(10))
```

Example output:
```
Elastic Net selected 3 of 6 variables
Selected variables: ['BMI', 'Diabetes', 'INPWT']

       Variable  Coefficient  Abs_Coefficient  Selected
0         INPWT       0.8234           0.8234      True
1      Diabetes       0.4521           0.4521      True
2           BMI       0.3012           0.3012      True
3    Sex_Female       0.0000           0.0000     False
4           Age       0.0000           0.0000     False
5  ASA_Class_II       0.0000           0.0000     False
```

### For Count Outcomes

```python
# For length of stay (count outcome)
selector = ElasticNetVariableSelection(
    df=cleaned_df,
    response_variable='Length_of_Stay',
    explanatory_variables=predictors,
    outcome_type='count',  # Use count regression
    alpha_ratio=0.5,
    cv=5
)
```

### Consensus Approach

Combine both methods to identify robust predictors:

```python
# Univariate selection
univariate = UnivariateVariableSelection(df, 'SSI', predictors, alpha=0.1)
univariate_selected = set(univariate.selected_explanatory_variables)

# Elastic net selection
elastic_net = ElasticNetVariableSelection(
    df, 'SSI', predictors, outcome_type='binary', alpha_ratio=0.5
)
elastic_net_selected = set(elastic_net.selected_explanatory_variables)

# Consensus: selected by both methods
consensus = univariate_selected & elastic_net_selected
print(f"Consensus variables: {consensus}")

# Union: selected by either method
union = univariate_selected | elastic_net_selected
print(f"Union variables: {union}")
```

## Statistical Tests by Variable Type

### Continuous Outcome

| Predictor Type | Test | Null Hypothesis |
|----------------|------|-----------------|
| Continuous | Pearson correlation | No linear association |
| Binary | Independent t-test | Equal means |
| Multi-category | One-way ANOVA | Equal means across groups |

### Categorical Outcome

| Predictor Type | Test | Null Hypothesis |
|----------------|------|-----------------|
| Continuous | T-test or ANOVA | Equal means across outcome groups |
| Categorical (2x2, small n) | Fisher's exact | Independence |
| Categorical (other) | Chi-square | Independence |

## Selection Strategy Comparison

### Univariate Testing

**Advantages:**
- Simple and interpretable
- Tests each association individually
- Standard in clinical research
- No assumptions about model form

**Limitations:**
- Ignores confounding
- Ignores interactions
- May select collinear variables
- Multiple testing increases false positives

**Best for:**
- Initial screening
- Descriptive analysis
- Small sample sizes
- When clinical interpretation of each association is important

### Elastic Net

**Advantages:**
- Accounts for multicollinearity
- Handles p >> n (more predictors than observations)
- Cross-validation prevents overfitting
- Combines feature selection and prediction

**Limitations:**
- Less interpretable (shrinkage)
- Requires sufficient sample size for CV
- May miss important confounders with weak marginal effects
- Sensitive to scaling

**Best for:**
- High-dimensional data
- Correlated predictors
- Prediction-focused analysis
- When parsimony is important

## Alpha Parameter Selection

### For Univariate Testing

**alpha=0.05:** Conservative, reduces false positives
- Risk: May miss important confounders
- Use when: Sample size is large, few candidate variables

**alpha=0.10:** Liberal, common for screening (default)
- Risk: More false positives, but better confounding control
- Use when: Initial screening, moderate sample size

**alpha=0.20:** Very liberal, used by some for initial models
- Risk: Many false positives
- Use when: Small sample, exploratory analysis

### For Elastic Net (alpha_ratio)

**alpha_ratio=0.0:** Pure Ridge (L2 only)
- No variable selection
- All coefficients shrunk but non-zero
- Best for: Highly collinear predictors

**alpha_ratio=0.5:** Balanced (default)
- Variable selection + multicollinearity handling
- Good general-purpose choice
- Best for: Most clinical applications

**alpha_ratio=1.0:** Pure Lasso (L1 only)
- Aggressive variable selection
- Sparse solutions
- Best for: When true model is sparse

## Recommended Workflow

1. **Start with univariate (alpha=0.1)** for initial screening
2. **Run elastic net (alpha_ratio=0.5)** for multicollinearity handling
3. **Identify consensus variables** (selected by both)
4. **Add clinically important variables** even if not selected
5. **Fit final models** with unadjusted, univariate-adjusted, elastic net-adjusted, and consensus-adjusted covariate sets
6. **Compare results** to assess robustness