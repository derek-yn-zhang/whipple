# API Reference

This section documents the `statwise` library used for statistical analysis in this project. The library provides a modular, reproducible pipeline for clinical data analysis.

## Overview

The `statwise` package consists of five main modules that work together to process raw data through to final statistical models:

```{mermaid}
graph LR
    A[Raw CSV] --> B[loading.py]
    B --> C[cleaning.py]
    C --> D[preparation.py]
    D --> E[selection.py]
    E --> F[modeling.py]
    F --> G[Results]
```

### Module Descriptions

**{doc}`loading`**
: Intelligent CSV loading with automatic type inference. Handles datetime parsing, integer preservation, and categorical conversion.

**{doc}`cleaning`**
: Modular data cleaning pipeline for reproducible preprocessing. Removes problematic variables and creates derived features.

**{doc}`preparation`**
: Transforms cleaned data into model-ready format. Handles encoding, standardization, and separation checking.

**{doc}`selection`**
: Variable selection using univariate testing and elastic net regularization. Identifies confounders and predictive features.

**{doc}`modeling`**
: Statistical modeling with logistic and negative binomial regression. Provides convergence diagnostics and interpretable results.

## Design Principles

The library follows best practices for clinical research:

1. **Reproducibility**: All operations are logged and deterministic
2. **Transparency**: Clear documentation of statistical decisions
3. **Modularity**: Each step can be inspected and modified independently
4. **Clinical Focus**: Methods aligned with clinical research standards (10 EPV rule, separation checking, etc.)

## Quick Start

```python
from statwise.loading import CSVDataLoader
from statwise.cleaning import DataCleaner
from statwise.preparation import DataPreparer
from statwise.selection import UnivariateVariableSelection
from statwise.modeling import LogisticRegressionModel

# Load data
loader = CSVDataLoader('data.csv')
df, _ = loader.load()

# Clean data
cleaner = DataCleaner(df, column_metadata)
cleaned_df, _ = cleaner.run_cleaning_pipeline(...)

# Prepare for modeling
preparer = DataPreparer(cleaned_df, 'outcome', ['var1', 'var2'])
X, y = preparer.preprocess()

# Select variables
selector = UnivariateVariableSelection(cleaned_df, 'outcome', predictors)
selected_vars = selector.selected_explanatory_variables

# Fit model
model = LogisticRegressionModel()
model.fit(X, y)
print(model.summary())
```