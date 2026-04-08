# Contributing

This repository is tutorial-first, so code readability matters almost as much as correctness. The `src/fomocid` package is the reusable layer underneath the notebooks and scripts, and contributors should document it in a way that helps both API readers and tutorial readers move quickly.

## Generated Files Policy

This repository is source-only. Do not commit generated artifacts such as:

- `docs/_build/`
- run outputs under `outputs/`
- `__pycache__/` directories
- `*.egg-info/`
- generated checkpoints or embedding bundles

If you build docs or run training locally, keep those artifacts untracked and rely on `.gitignore` to enforce that boundary.

## Docstring Style

We use **Google-style docstrings** throughout `src/fomocid`, formatted for Sphinx with `sphinx.ext.napoleon`.

Use this pattern consistently:

```python
def example(arg: int, enabled: bool = True) -> str:
    """Short one-line summary.

    Args:
        arg: What the argument means in the context of the function.
        enabled: What changes when the flag is enabled.

    Returns:
        Description of the returned value.

    Raises:
        ValueError: When invalid inputs are provided.
    """
```

Guidelines:

- Start with a short imperative or descriptive summary sentence.
- Document public modules, classes, functions, methods, and non-obvious helpers.
- Public callables with parameters should include an `Args:` section.
- Use `Returns:` and `Raises:` whenever they add useful structure.
- Prefer describing behavior and intent over repeating type annotations verbatim.
- Keep parameter descriptions concise, but explain defaults when they change behavior.
- Mention shapes for tensors when that helps the reader understand the contract.
- For internal helpers, add a docstring when the name alone does not make the behavior obvious.

## What To Document

Add docstrings when you introduce or significantly change:

- public functions exported from package `__init__.py` files
- Lightning modules and their training/evaluation hooks
- typed config interfaces in `fomocid.config_types`
- configuration and IO helpers used by the tutorial scripts
- notebook utility functions that generate visuals or parse artifacts
- analysis helpers that expect specific tensor shapes or normalization schemes

Short wrappers can stay brief, but they should still explain why they exist when that is not already obvious from the signature.

## API and Tutorial Relationship

The notebooks are the primary teaching surface, but the reusable contracts live in `src/fomocid`. Good docstrings help in three places at once:

- the source code stays self-explaining for contributors
- the generated Sphinx API reference is useful instead of skeletal
- notebook authors can quickly discover helper behavior without tracing the full implementation

## Sphinx Notes

The docs build reads these docstrings directly into the API reference. To keep that output clean:

- avoid empty placeholder docstrings
- keep the first line short enough to work as a summary
- prefer plain English over implementation trivia
- include tensor shapes only when they clarify the API contract
- replace vague phrases like "configuration dictionary" with the actual section or key names that matter

You can preview the docs locally with:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Contribution Checklist

Before opening a change that touches `src/fomocid`:

- add or update docstrings for the affected public API
- keep `fomocid.config_types` aligned with the YAML config contract
- keep the wording in Google style
- make sure the docs build still succeeds
- update user-facing docs if the behavior or configuration surface changed
