# deep-ml

[Deep-ML](https://www.deep-ml.com/) resolutions

## Quick Start

### 1. Create an virtual enviroment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install the dependencies
```bash
pip install -U pip && pip install -r requirements.txt
# or
make install
```

### 3. To execute all the resolutions tests:
```bash
pytest -vvs .
# or
make tests
```

### 4. To format the code:
```bash
ruff check --select I --fix && ruff format
# or
make fmt
```
