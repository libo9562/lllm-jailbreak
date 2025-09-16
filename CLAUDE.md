# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of the "DeepInception" paper, which studies jailbreaking vulnerabilities in large language models for defensive security research. The codebase implements a method to test LLM safety guardrails by leveraging personification abilities to create nested scenarios.

## Development Commands

### Setup and Dependencies
```bash
# Install dependencies (requires Python >=3.12)
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Code Quality
```bash
# Run all quality checks, linting, formatting, and tests
./scripts/update.sh

# Individual commands (use update.sh instead):
# Format imports
isort .

# Lint and format code
ruff check .
ruff format .
```

### Pre-commit Checks
**IMPORTANT**: Always run `./scripts/update.sh` before committing. This script runs:
- `no_implicit_optional` for type checking
- `autoimport` for import optimization
- `ruff check --fix` for linting with auto-fixes
- `ruff format` for code formatting
- `pytest --cov` for tests with coverage

### Running Experiments
```bash
# Basic test run
python main.py --exp_name test --target-model llama3.2:1b

# Main experiment
python main.py --exp_name main --target-model llama3.2:1b

# With defense mechanisms
python main.py --exp_name test --defense sr  # Self-Reminder defense
python main.py --exp_name test --defense ic  # In-Context defense
```

## Architecture

### Core Components

- **main.py**: Entry point that orchestrates experiment execution, loads data from JSON files in `res/` directory, and processes questions through target models
- **conversers.py**: Model factory that creates LangChain-based model wrappers, currently supports Ollama provider with ChatOllama

### Data Flow

1. Experiment data is loaded from `res/data_{exp_name}.json` files
2. Target model is initialized via `load_models()` factory function
3. Questions are processed sequentially through the model chain
4. Results are saved to `results/{model}_{exp_name}_{defense}_results.json`

### Experiment Types

The codebase supports multiple experiment configurations via `--exp_name`:
- `test`: Basic testing with inception attack + questions
- `main`: Main experiment setup
- `abl_c`, `abl_layer`: Ablation studies
- `multi_scene`: Multiple scenario testing
- `abl_fig6_4`, `further_q`: Additional experimental variants

### Model Integration

Uses LangChain framework for model abstraction with Ollama as the current provider. The model wrapper handles:
- Token limit configuration (`target_max_n_tokens`)
- Temperature settings for response variability
- Output parsing through `StrOutputParser`

## Required Data Files

The system expects JSON data files in the `res/` directory with the naming pattern `data_{exp_name}.json`. Each data file should contain experiment-specific questions and scenarios for the jailbreaking tests.