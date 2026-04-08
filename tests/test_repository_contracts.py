from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "fomocid"
REFERENCE_PATTERN = re.compile(r"(?P<path>(?:configs|docs|tutorials)/[A-Za-z0-9_./-]+\.(?:yaml|md|ipynb|py))")


class RepositoryContractTests(unittest.TestCase):
    def test_documented_local_paths_exist(self) -> None:
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

    def test_public_source_modules_have_docstrings(self) -> None:
        for module_path in sorted(PACKAGE_ROOT.rglob("*.py")):
            if "__pycache__" in module_path.parts:
                continue
            module = ast.parse(module_path.read_text(encoding="utf-8"))
            module_docstring = ast.get_docstring(module)
            self.assertTrue(module_docstring, msg=f"Missing module docstring in {module_path.relative_to(REPO_ROOT)}")

            for node in module.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                    self._assert_docstring_with_args(node, module_path)
                if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    class_docstring = ast.get_docstring(node)
                    self.assertTrue(class_docstring, msg=f"Missing class docstring for {node.name} in {module_path.relative_to(REPO_ROOT)}")
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__":
                            self._assert_docstring_with_args(child, module_path, owner=node.name)

    def _assert_docstring_with_args(self, node: ast.FunctionDef | ast.AsyncFunctionDef, module_path: Path, owner: str | None = None) -> None:
        docstring = ast.get_docstring(node)
        name = f"{owner}.{node.name}" if owner else node.name
        self.assertTrue(docstring, msg=f"Missing docstring for {name} in {module_path.relative_to(REPO_ROOT)}")
        positional_args = [arg.arg for arg in node.args.args if arg.arg not in {"self", "cls"}]
        if positional_args:
            self.assertIn("Args:", docstring, msg=f"Missing Args section for {name} in {module_path.relative_to(REPO_ROOT)}")
