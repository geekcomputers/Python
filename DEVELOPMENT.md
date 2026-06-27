# Development Guide

Welcome to geek-computers Python repository! This guide will help you set up your development environment and contribute to the project.

## Prerequisites

- Python 3.10+ (we test against 3.10, 3.11, and 3.12)
- pip (Python package manager)
- git

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/geekcomputers/Python.git
cd Python
```

### 2. Create Virtual Environment
```bash
# On Linux/Mac
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Development Tools
```bash
pip install ruff bandit mypy pytest codespell
```

## Code Quality Checks

Before submitting a PR, run these checks locally:

### Run Codespell (spell checking)
```bash
codespell --skip "*.json,*.txt,*.pdf,*.md"
```

### Run Bandit (security analysis)
```bash
bandit -r . --skip B101,B105
```

### Run Ruff (linting)
```bash
ruff check .
```

### Run Pytest (unit tests)
```bash
pytest
```

### Run MyPy (type checking)
```bash
mypy . --ignore-missing-imports
```

## Writing Scripts

When adding new scripts:

1. **Use clear, descriptive names** - `calculate_fibonacci.py` instead of `test.py`
2. **Add docstrings** - Describe what your script does
3. **Include comments** - Explain complex logic
4. **Use type hints** (where applicable) - Helps with code clarity
5. **Test your code** - Run it locally before submitting

### Example Script Template
```python
"""
Module Name: Calculate Fibonacci Sequence
Description: Generate Fibonacci numbers up to N
Author: Your Name
Date: YYYY-MM-DD
"""

def fibonacci(n: int) -> list[int]:
    """
    Generate Fibonacci sequence up to nth number.
    
    Args:
        n: Number of Fibonacci numbers to generate
        
    Returns:
        List of Fibonacci numbers
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib


if __name__ == "__main__":
    result = fibonacci(10)
    print(result)
```

## Submission Process

1. **Create a feature branch** - `git checkout -b feature/your-feature-name`
2. **Make your changes** - Add or modify scripts
3. **Run quality checks** - Use the commands above
4. **Commit your changes** - Use clear commit messages
5. **Push to your fork** - `git push origin feature/your-feature-name`
6. **Create a Pull Request** - Include description and testing info

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Update README.md if adding new scripts
- Reference any related issues with "Closes #123"
- Follow the pull request template

## Common Issues

### ImportError for dependencies
Make sure you've activated the virtual environment and installed all requirements:
```bash
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Tests fail locally but not in CI
Check your Python version - we test on 3.10, 3.11, and 3.12:
```bash
python --version
```

### Code style issues
Run ruff with auto-fix:
```bash
ruff check . --fix
```

## Need Help?

- Check existing [issues](https://github.com/geekcomputers/Python/issues)
- Review the [README.md](README.md) for script documentation
- Email: craig@geekcomputers.co.uk

Happy coding! 🐍
