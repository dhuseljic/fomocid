"""Test repository documentation and path-reference contracts."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATTERN = re.compile(r"(?P<path>(?:configs|docs|tutorials)/[A-Za-z0-9_./-]+\.(?:yaml|md|ipynb|py))")
PYTHON_DOCSTRING_ROOTS = (
    REPO_ROOT / "src",
    REPO_ROOT / "tests",
    REPO_ROOT / "tutorials",
)


class RepositoryContractTests(unittest.TestCase):
    def test_documented_local_paths_exist(self) -> None:
        """Test that documented local paths exist.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        sources = [
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs" / "index.md",
            REPO_ROOT / "docs" / "tutorials.md",
            REPO_ROOT / "docs" / "configuration.md",
            REPO_ROOT / "tests" / "test_smoke.py",
            REPO_ROOT / "tutorials" / "train_ssl.py",
            REPO_ROOT / "tutorials" / "evaluate_representations.py",
        ]
        for source_path in sources:
            text = source_path.read_text(encoding="utf-8")
            for match in REFERENCE_PATTERN.finditer(text):
                referenced_path = REPO_ROOT / match.group("path")
                self.assertTrue(
                    referenced_path.exists(),
                    msg=f"{source_path.relative_to(REPO_ROOT)} references missing path {match.group('path')}",
                )

    def test_python_functions_have_numpy_style_docstrings(self) -> None:
        """Test that Python functions have NumPy-style docstrings.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        for root in PYTHON_DOCSTRING_ROOTS:
            for module_path in sorted(root.rglob("*.py")):
                if "__pycache__" in module_path.parts:
                    continue
                module = ast.parse(module_path.read_text(encoding="utf-8"))
                for node in ast.walk(module):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._assert_docstring_with_args(node, module_path)

    def test_python_files_have_module_summaries(self) -> None:
        """Test that Python files have module summaries.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        for root in PYTHON_DOCSTRING_ROOTS + (REPO_ROOT / "docs",):
            for module_path in sorted(root.rglob("*.py")):
                if "__pycache__" in module_path.parts or ".egg-info" in module_path.parts:
                    continue
                module = ast.parse(module_path.read_text(encoding="utf-8"))
                module_docstring = ast.get_docstring(module)
                self.assertTrue(
                    module_docstring,
                    msg=f"Missing module docstring in {module_path.relative_to(REPO_ROOT)}",
                )

    def _assert_docstring_with_args(self, node: ast.FunctionDef | ast.AsyncFunctionDef, module_path: Path, owner: str | None = None) -> None:
        """Assert that a function has description, parameters, and returns.

        Parameters
        ----------
        node : ast.FunctionDef | ast.AsyncFunctionDef
            Input value for ``node``.
        module_path : Path
            Input value for ``module_path``.
        owner : str | None
            Input value for ``owner``.

        Returns
        -------
        None
            The function completes in place.
        """
        docstring = ast.get_docstring(node)
        name = f"{owner}.{node.name}" if owner else node.name
        self.assertTrue(docstring, msg=f"Missing docstring for {name} in {module_path.relative_to(REPO_ROOT)}")
        self.assertIn(
            "Parameters",
            docstring,
            msg=f"Missing Parameters section for {name} in {module_path.relative_to(REPO_ROOT)}",
        )
        self.assertTrue(
            "Returns" in docstring or "Yields" in docstring,
            msg=f"Missing Returns section for {name} in {module_path.relative_to(REPO_ROOT)}",
        )
