#!/bin/bash

# Use provided arguments, otherwise default to 'src' and 'tests'
if [ "$#" -eq 0 ]; then
    DIRS=("src" "tests")
else
    DIRS=("$@")
fi

# Run commands for each directory
uv run no_implicit_optional "${DIRS[@]}"
uv run autoimport "${DIRS[@]}"
uv run ruff check --fix "${DIRS[@]}"
uv run ruff format "${DIRS[@]}"

set -e
# Run unit tests and check coverage
uv run pytest --cov-config=./pyproject.toml --cov=src tests
