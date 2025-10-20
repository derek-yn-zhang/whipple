# Whipple Surgery INPWT Analysis

Statistical analysis of incisional negative pressure wound therapy (INPWT) outcomes in pancreatic surgery (Whipple procedure).

## Quick Start

### 1. Install UV (one-time setup)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installation.

### 2. Run the Analysis
```bash
# Clone the repository
git clone https://github.com/derek-yn-zhang/whipple.git
cd whipple

# Launch Jupyter (UV handles everything automatically)
uv run jupyter lab
```

**That's it!** UV will:
- Install Python 3.10.13 automatically
- Install all required packages
- Launch Jupyter Lab

### 3. Open `analysis.ipynb` and Run

Click **"Run" → "Run All Cells"** to reproduce the complete analysis.

---

## Project Structure
```
whipple/
├── statwise/             # Statistical analysis library
│   ├── loading.py        # CSV data loading with type inference
│   ├── cleaning.py       # Data cleaning and preprocessing
│   ├── preparation.py    # Model-ready data preparation
│   ├── selection.py      # Variable selection methods
│   ├── modeling.py       # Regression models
│   └── results.py        # Results extraction and reporting
├── analysis.ipynb        # Main analysis notebook
├── data/                 # Data files
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

---

## Methodology

### Variable Selection
- **Univariate testing** (p < 0.1): Classical statistical screening
- **Elastic net regularization** (L1 ratio = 0.5): Handles multicollinearity
- **Consensus approach**: Variables selected by both methods

### Sample Size Considerations
- **Binary outcomes**: 10 events per variable (Peduzzi et al., 1996)
- **Count outcomes**: 15 observations per variable (conservative threshold)

### Models
- **Logistic regression**: Binary outcomes (sepsis, SSI, readmission)
- **Negative binomial regression**: Count outcomes (length of stay)

### Four-Model Approach
For each outcome:
1. Unadjusted (treatment only)
2. Univariate-adjusted
3. Elastic net-adjusted
4. Consensus-adjusted

---

## For Developers

### Run Tests
```bash
uv run pytest
```

### Build Documentation
```bash
uv run --extra docs jupyter-book build book/
```

### Add a Package
```bash
uv add package-name
```

### Development Mode
```bash
uv run --extra dev pytest  # With dev dependencies
```

---

## Requirements

- **No manual setup required** - UV handles everything
- Python 3.10+ (automatically installed by UV)
- ~500MB disk space for dependencies

---

## License

MIT License - see LICENSE file for details.

---

## Contact

For questions about the methodology or code, please open an issue.