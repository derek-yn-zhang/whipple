# Data Loading

The `loading` module provides intelligent CSV reading with automatic type inference and normalization.

## CSVDataLoader

```{eval-rst}
.. autoclass:: statwise.loading.CSVDataLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Basic Loading

```python
from statwise.loading import CSVDataLoader

loader = CSVDataLoader('surgical_data.csv')
df, log_messages = loader.load()

print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
print(df.dtypes)
```

### Custom CSV Parameters

```python
# Handle different encodings or delimiters
loader = CSVDataLoader('data.csv')
df, logs = loader.load(
    encoding='latin1',
    sep=';',
    na_values=['NA', 'Missing', '']
)
```

### Type Inference Results

The loader automatically:
- Converts date-like strings to datetime64
- Preserves integers as nullable Int64 (supports missing values)
- Converts all object columns to categorical for memory efficiency

Example output:
```
=== Loading CSV file: surgical_data.csv ===
Initial shape: (165, 45)
Parsed date/time columns: ['Surgery_Date', 'Discharge_Date']
Converted to Int64 (nullable integer): ['Age', 'Length_of_Stay', 'Num_Complications']
Converted 12 object columns to categorical.
=== CSV Loading Complete ===
```

## Design Rationale

### Why Nullable Int64?

Standard pandas reads integers with missing values as float64, which can be misleading for count data. The nullable Int64 dtype preserves the integer nature while supporting NA values.

### Why Categorical?

Categorical dtype reduces memory usage by 50-90% for columns with repeated values (common in clinical data like race, procedure type, comorbidities).

### Why Automatic Date Parsing?

Dates are detected by sampling values and checking if >80% parse successfully. This avoids false positives while catching true date columns.