# Data Cleaning

The `cleaning` module provides a modular pipeline for reproducible data preprocessing with explicit statistical power considerations.

## DataCleaner

```{eval-rst}
.. autoclass:: statwise.cleaning.DataCleaner
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Full Cleaning Pipeline

```python
from statwise.cleaning import DataCleaner

# Define column roles
metadata = {
    'SSI': 'response',
    'Sepsis': 'response',
    'Length_of_Stay': 'response',
    'Age': 'explanatory',
    'Sex': 'explanatory',
    'BMI': 'explanatory',
    'INPWT': 'explanatory'
}

cleaner = DataCleaner(df, metadata)

cleaned_df, logs = cleaner.run_cleaning_pipeline(
    drop_cols=['Patient_ID', 'MRN', 'Attending_Surgeon'],
    keep_datetime_cols=None,  # Drop all datetime columns
    keep_response_cols=['SSI', 'Sepsis', 'Length_of_Stay'],
    keep_explanatory_cols=['Age', 'Sex', 'BMI', 'Diabetes', 'INPWT'],
    min_minority_count=10,  # 10 events per variable rule
    max_missing_rate=0.40   # Max 40% missingness
)
```

### Deriving Composite Variables

```python
def create_any_ssi(df):
    """Any SSI: superficial, deep, or organ/space."""
    return ((df['Superficial_SSI'] == 'Yes') | 
            (df['Deep_SSI'] == 'Yes') | 
            (df['Organ_Space_SSI'] == 'Yes')).astype(int)

def standardize_bmi(df):
    """Convert weight and height to BMI in kg/m^2."""
    weight_kg = df['Weight_kg']
    height_m = df['Height_cm'] / 100
    return weight_kg / (height_m ** 2)

derivation_map = {
    'Any_SSI': create_any_ssi,
    'BMI': standardize_bmi
}

drop_components = {
    'Any_SSI': ['Superficial_SSI', 'Deep_SSI', 'Organ_Space_SSI'],
    'BMI': ['Weight_kg', 'Height_cm']
}

cleaner.derive_variables(derivation_map, drop_components)
```

### Cleaning Categorical Variables

```python
# Consolidate rare categories
consolidation_maps = {
    'Race': {
        'White': 'White',
        'Black or African American': 'Black',
        'Asian': 'Other',
        'Native Hawaiian or Pacific Islander': 'Other',
        'Some Other Race': 'Other'
    }
}

# Apply ordinal structure
ordered_maps = {
    'ASA_Class': [
        'ASA I - Normal healthy',
        'ASA II - Mild systemic disease',
        'ASA III - Severe systemic disease',
        'ASA IV - Severe disease, constant threat to life'
    ]
}

cleaner.clean_categorical_predictors(consolidation_maps, ordered_maps)
```

## Statistical Power Considerations

### The 10 Events Per Variable Rule

The default `min_minority_count=10` follows Peduzzi et al.'s simulation study showing that logistic regression requires approximately 10 events per predictor variable for reliable estimation.

For a binary outcome with 30 events:
- Maximum 3 predictors should be included
- Variables with <10 observations in the minority class are dropped

### The 40% Missingness Threshold

Variables with >40% missing data are dropped by default because:
- Multiple imputation becomes unreliable
- Complete case analysis loses too much power
- Remaining data may not be representative

### Categorical vs Continuous Integer Detection

The cleaner distinguishes between:
- **Quasi-categorical integers**: ≤10 unique values OR <10% uniqueness ratio (e.g., ASA class 1-5)
- **Continuous integers**: High cardinality (e.g., length of stay 0-100 days)

Only quasi-categorical integers are checked for class imbalance.

## Design Philosophy

The cleaning pipeline emphasizes:
1. **Explicit decisions**: No silent data modifications
2. **Comprehensive logging**: Every operation is recorded
3. **Statistical validity**: Removes variables that compromise power
4. **Reproducibility**: Same inputs always produce same outputs