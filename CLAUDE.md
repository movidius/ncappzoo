# CLAUDE.md - Project Information for Claude Code

## Project Overview
Intel Neural Compute Application Zoo (ncappzoo) - Applications and network examples for Intel Neural Compute Stick.

## Key Commands
- Test command: `poetry run test` or `poetry run tests`
- Test with coverage: `poetry run pytest --cov`
- Run specific test markers: `poetry run pytest -m unit` or `poetry run pytest -m integration`
- Run tests without slow tests: `poetry run pytest -m "not slow"`
- Lint command: (to be determined)
- Type check command: (to be determined)

## Testing Infrastructure
- Testing framework: pytest with pytest-cov and pytest-mock
- Package manager: Poetry
- Test directories: tests/unit/ and tests/integration/
- Coverage threshold: 80%
- Test markers: unit, integration, slow

## Important Notes
- Testing infrastructure has been set up with Poetry package manager
- Run `poetry install` to install dependencies before running tests
- Coverage reports are generated in htmlcov/ directory and coverage.xml

