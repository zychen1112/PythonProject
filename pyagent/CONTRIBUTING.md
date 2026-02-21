# Contributing to PyAgent

Thank you for your interest in contributing to PyAgent! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

Be respectful and inclusive. We welcome contributions from everyone.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Setup Steps

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/pyagent.git
   cd pyagent
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies**

   ```bash
   pip install -e ".[dev]"
   pip install -e ".[openai,anthropic]"  # Optional: for provider testing
   ```

4. **Set up pre-commit hooks (optional)**

   ```bash
   pip install pre-commit
   pre-commit install
   ```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/zychen1112/PythonProject/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Wait for discussion before implementing

### Submitting Code

1. Create a feature branch from `main`
2. Make your changes
3. Add/update tests
4. Submit a pull request

## Code Style

We use the following tools to maintain code quality:

- **Ruff**: Linting and formatting
- **MyPy**: Type checking
- **Pytest**: Testing

### Running Checks

```bash
# Lint code
ruff check src/

# Format code
ruff format src/

# Type check
mypy src/

# Run tests
pytest

# Run tests with coverage
pytest --cov=pyagent
```

### Code Guidelines

- Use type hints for all public functions
- Write docstrings for classes and public methods
- Keep functions focused and under 50 lines when possible
- Follow PEP 8 conventions

## Commit Guidelines

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

**Examples:**

```
feat: add support for custom tool validators
fix: handle timeout in MCP client correctly
docs: update README with new installation options
test: add tests for skill loader
```

## Pull Request Process

1. **Create a branch**

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Run checks locally**

   ```bash
   ruff check src/
   ruff format src/
   mypy src/
   pytest
   ```

4. **Commit and push**

   ```bash
   git add .
   git commit -m "feat: your feature description"
   git push origin feat/your-feature-name
   ```

5. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure CI passes

6. **Code Review**
   - Respond to feedback promptly
   - Make requested changes
   - Squash commits if requested

## Need Help?

- Open a [Discussion](https://github.com/zychen1112/PythonProject/discussions)
- Ask in issues with the `question` label

Thank you for contributing to PyAgent!
