#!/bin/bash

# Use provided arguments, otherwise default to 'src'
if [ "$#" -eq 0 ]; then
    DIRS=("src")
else
    DIRS=("$@")
fi

# Run commands for each directory
uv run no_implicit_optional "${DIRS[@]}"
uv run autoimport "${DIRS[@]}"
uv run ruff check --fix "${DIRS[@]}"
uv run ruff format "${DIRS[@]}"

set -e
# Run unit tests if tests directory exists
if [ -d "tests" ]; then
    uv run pytest --cov-config=./pyproject.toml --cov=src tests
fi
