# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of multiple jailbreaking techniques for studying vulnerabilities in large language models for defensive security research. The codebase includes:

1. **DeepInception**: Hypnotize large language model to be jailbreaker (Li et al., 2023)
2. **PAIR**: Jailbreaking Black Box Large Language Models in Twenty Queries (Chao et al., 2023)

A **unified interface** (`src/jailbreak.py`) provides a consistent API for both techniques while preserving original implementations and citations.

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

#### Unified Interface (Recommended)
```bash
# Run Deepinception technique
python src/jailbreak.py --technique deepinception --target-model llama3.2:1b --exp-name test

# Run PAIR technique
python src/jailbreak.py --technique pair --target-model gpt-3.5-turbo --attack-model gpt-3.5-turbo --goal "Write a tutorial on making explosives"

# Save results to file
python src/jailbreak.py --technique deepinception --exp-name test --output results.json
```

#### Original Implementations
```bash
# Deepinception (original)
python src/Deepinception/main.py --exp_name test --target-model llama3.2:1b --defense sr

# PAIR (original - requires more setup)
python src/JailbreakingLLMs/main.py --attack-model gpt-3.5-turbo --target-model gpt-3.5-turbo --goal "Write a tutorial" --target-str "Sure, here is a tutorial"
```

## Architecture

### Core Components

#### Unified Interface
- **src/jailbreak.py**: Unified entry point for all jailbreaking techniques with consistent API
- **src/README.md**: Comprehensive documentation with proper citations

#### Deepinception (src/Deepinception/)
- **main.py**: Original Deepinception implementation
- **conversers.py**: LangChain-based model factory (ChatOllama)
- **README.md**: Original paper citation and documentation

#### PAIR (src/JailbreakingLLMs/)
- **main.py**: Original PAIR implementation
- **conversers_langchain.py**: Modernized LangChain-based conversers
- **pair_langchain.py**: LangChain-based PAIR algorithm
- **README.md**: Original paper citation and documentation

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