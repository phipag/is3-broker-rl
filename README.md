# `python-starter`

This is my project structure for new Python projects. It uses the following tools to automate the development process:

* `poetry`: Reproducible dependency management and packaging
* `tox`: Test and workflow automation against different Python environments
* `pytest`: Unittests
* `black`: Code formatting
* `isort`: Code imports formatting
* `flake8`: Code linting
* `mypy`: Static type checking

My goal was to provide an easy to start setup with all development good practices implemented.

## Where is my `setup.py` / `setup.cfg`?

I intentionally did not provide a `setup.py` / `setup.cfg` configuration because this package is managed by Poetry's
build system. This follows [PEP 517](https://www.python.org/dev/peps/pep-0517/) where `setuptools` is no longer the
default build system for Python. Instead it is possible to configure build systems via `pyproject.toml`. Poetry provides
a consistent configuration solely based on the `pyproject.toml` file. Read more details
here: https://setuptools.pypa.io/en/latest/build_meta.html.

## Usage

### Installation

Install [`poetry`](https://python-poetry.org/) on your machine if not yet installed. Install the project dependencies
using:

```shell
poetry install
```

### Development process

You can invoke all workflows using the `tox` CLI:

```shell
tox -e format # Format code
tox -e lint # Lint code
tox -e format,lint # Format first and then lint
tox -e python3.10 # Run pytest tests for Python 3.10 environment

# Run all workflows in logical order:
# format -> lint -> pytest against all Python environments
tox
```

### Where to find the Python interpreter path?

It might be useful to find the Python interpreter path for integration with your IDE. Print the path using:

```shell
poetry env info --path
```

For usage from command-line directly run:

```shell
poetry run <command> # Will run any command within the Poetry Python virtual environment
poetry shell # Will start a new shell as child process (recommended)
source `poetry env info --path`/bin/activate # Activation in current shell
```