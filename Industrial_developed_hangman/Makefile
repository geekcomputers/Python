lint:
	poetry run isort src tests
	poetry run flake8 src tests
	poetry run mypy src
	poetry run mypy tests

test:
	poetry run pytest

build:
	python src/hangman/main.py
install:
	pip install poetry
	poetry install