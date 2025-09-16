#!/bin/bash

# Use provided arguments, otherwise default to 'src' and 'tests'
if [ "$#" -eq 0 ]; then
    DIRS=("src" "tests")
else
    DIRS=("$@")
fi

# Run commands for each directory
poetry run no_implicit_optional "${DIRS[@]}"
poetry run autoimport "${DIRS[@]}"
poetry run ruff check --fix "${DIRS[@]}"
poetry run ruff format "${DIRS[@]}"

set -e
# Run unit tests and check coverage
poetry run pytest --cov-config=./pyproject.toml --cov=src tests
