"""
Statwise: Statistical Analysis Pipeline for Clinical Research
==============================================================

A modular framework for reproducible clinical outcomes analysis with
emphasis on small-sample robustness and transparent reporting.

Modules
-------
loading
    CSV data loading with automatic type inference
cleaning
    Systematic data cleaning and preprocessing
preparation
    Model-ready data preparation and separation checking
selection
    Variable selection via univariate testing and elastic net
modeling
    Regression models with convergence handling
results
    Automated results extraction and reporting

Examples
--------
Basic workflow:

>>> from statwise.loading import CSVDataLoader
>>> from statwise.cleaning import DataCleaner
>>> from statwise.modeling import run_analysis
>>>
>>> # Load data
>>> loader = CSVDataLoader('data.csv')
>>> df, logs = loader.load()
>>>
>>> # Clean data
>>> cleaner = DataCleaner(df, column_metadata)
>>> df_clean, logs = cleaner.run_cleaning_pipeline(...)
>>>
>>> # Run analysis
>>> results = run_analysis(df_clean, response_vars, explanatory_vars)
"""

__version__ = "0.1.0"

from .loading import CSVDataLoader
from .cleaning import DataCleaner
from .preparation import DataPreparer
from .selection import UnivariateVariableSelection, ElasticNetVariableSelection
from .modeling import (
    LogisticRegressionModel,
    NegativeBinomialRegressionModel,
)

__all__ = [
    "CSVDataLoader",
    "DataCleaner",
    "DataPreparer",
    "UnivariateVariableSelection",
    "ElasticNetVariableSelection",
    "LogisticRegressionModel",
    "NegativeBinomialRegressionModel",
]
