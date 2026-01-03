# Language Style Matching for Japanese

This repository implements Language Style Matching (LSM) and reciprocal Language Style Matching (rLSM) algorithms for Japanese text analysis.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd LanguageStyleMatching
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

This will install all required packages including:
- pandas, numpy (data processing)
- spacy, ja-ginza (Japanese NLP)
- pytest, pytest-cov, pytest-mock (testing)

## Testing

This repository includes comprehensive test suites for both LSM and rLSM implementations.

### Running All Tests

```bash
pytest -v
```

### Running Specific Test Modules

```bash
# LSM tests only
pytest lsm/tests/ -v

# rLSM tests only
pytest rlsm/tests/ -v

# Specific test file
pytest lsm/tests/test_core.py -v
```

### Test Coverage

To generate a coverage report:
```bash
pytest --cov=lsm --cov=rlsm --cov-report=html
```

### Expected Results

All 31 tests should pass:
- LSM core functionality tests
- rLSM core functionality tests
- Paper example validations (Müller-Frommeyer et al. 2019)
- Worker and runner tests

## References

[1] Gonzales, Amy L., Jeffrey T. Hancock, and James W. Pennebaker. 2010. "Language Style Matching as a Predictor of Social Dynamics in Small Groups." Communication Research 37 (1): 3–19.

[2] Müller-Frommeyer, Lena C., Niels A. M. Frommeyer, and Simone Kauffeld. 2019. "Introducing rLSM: An Integrated Metric Assessing Temporal Reciprocity in Language Style Matching." Behavior Research Methods 51 (3): 1343–59.

[3] Khaleghzadegan, Salar, Michael Rosen, Anne Links, Alya Ahmad, Molly Kilcullen, Emily Boss, Mary Catherine Beach, and Somnath Saha. 2024. "Validating Computer-Generated Measures of Linguistic Style Matching and Accommodation in Patient-Clinician Communication." Patient Education and Counseling 119 (108074): 108074.

[4] Igarashi, Tasuku, Shimpei Okuda, and Kazutoshi Sasahara. 2022. "Development of the Japanese Version of the Linguistic Inquiry and Word Count Dictionary 2015." Frontiers in Psychology 13 (March): 841534.
