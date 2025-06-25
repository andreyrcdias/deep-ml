install:
	pip install -U pip
	pip install -r requirements.txt

fmt:
	ruff check --select I --fix
	ruff format

tests:
	pytest -vvs .

