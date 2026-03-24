# AGENT.md

## Project Initialization Rules

- Use the `src/` layout for all production Python code.
- Keep automated tests under `tests/`.
- Put architecture notes, onboarding guides, and design docs under `docs/`.
- Store non-secret runtime and tool configuration templates under `config/`.
- Keep the supported Python version aligned across `.python-version`, `pyproject.toml`, CI, and local tooling.
- Create and use a local virtual environment at `.venv/`.
- Commit only source files and configuration; never commit virtual environments, caches, secrets, or build artifacts.

## Python Coding Standards

### Style Guide

- Follow PEP 8 and keep line length at 88 characters unless a file already follows a different enforced project convention.
- Use `ruff format` formatting conventions for whitespace, quotes, and import layout.
- Prefer small, single-purpose functions and classes over monolithic modules.
- Avoid hidden side effects at import time.

### Naming Conventions

- Modules and packages: `snake_case`
- Variables and functions: `snake_case`
- Classes and exceptions: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers: prefix with a single leading underscore
- Type variables: descriptive `PascalCase` names ending in `T` only when that improves clarity

### Code Organization

- Keep production code inside `src/hello_agents/`.
- Group modules by domain responsibility, not by technical layer alone.
- Keep `__init__.py` files minimal and avoid re-exporting large APIs without a clear need.
- Move shared helpers into focused modules instead of circular cross-imports.
- Put test fixtures close to the tests that use them; promote them to shared fixtures only when reused.

### Docstring Rules

- Write docstrings for all public modules, classes, and functions.
- Use triple double quotes.
- Start with a short imperative summary line.
- Add sections such as `Args`, `Returns`, `Raises`, and `Examples` when they improve clarity.
- Keep internal private helpers lightly documented unless the logic is non-obvious.

### Type Annotations

- Add type annotations to all public functions, methods, and module-level constants where practical.
- New or modified functions should include full parameter and return annotations.
- Prefer standard library typing features and modern built-in generics.
- Use `TypedDict`, `Protocol`, `Enum`, or dataclasses when they clarify structured data contracts.
- Avoid `Any` unless there is a clear interoperability reason and it is documented.

### Exception Handling

- Raise specific exception types instead of bare `Exception`.
- Do not swallow exceptions silently.
- Catch exceptions at boundaries where the code can add context, translate errors, retry safely, or recover.
- Preserve the original exception with `raise ... from exc` when re-raising with additional context.
- Keep error messages actionable and precise.

### Import Rules

- Order imports as standard library, third-party, then local application imports.
- Use absolute imports inside the project unless a short relative import materially improves readability within a tightly scoped package.
- Avoid wildcard imports.
- Import only what is used.
- Resolve import cycles by restructuring modules instead of delaying imports except as a last resort.

## Quality Gate

- Run `pytest` for tests.
- Run `ruff check .` and `ruff format --check .` before merging.
- Run `mypy src` for static type checks on changed code paths.

