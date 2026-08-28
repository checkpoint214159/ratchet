"""AST checks for the declared bounded-context forbidden-import policy."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_CONTEXTS = frozenset(
    {
        "evaluation",
        "models",
        "backends",
        "measurement",
        "experiments",
        "dispatch",
        "optimization",
        "reporting",
    }
)
CROSS_CONTEXT_SUBMODULE_ALLOWLIST: frozenset[str] = frozenset()


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
            imports.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imports


def _context_files(context: str) -> list[Path]:
    return sorted((ROOT / "ratchet" / context).rglob("*.py"))


def _imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _assert_no_import_prefix(context: str, prefixes: tuple[str, ...]) -> None:
    offenders = {
        path.relative_to(ROOT): sorted(
            imported for imported in _imports(path) if imported.startswith(prefixes)
        )
        for path in _context_files(context)
    }
    offenders = {path: imports for path, imports in offenders.items() if imports}
    assert not offenders, offenders


def _assert_cross_context_imports_are_public(
    context: str, imported_modules: set[str]
) -> None:
    nested_imports = {
        module
        for module in imported_modules
        if len(parts := module.split(".")) >= 3
        and parts[0] == "ratchet"
        and parts[1] in PUBLIC_CONTEXTS
        and parts[1] != context
        and module not in CROSS_CONTEXT_SUBMODULE_ALLOWLIST
    }
    assert not nested_imports, nested_imports


@pytest.mark.parametrize(
    ("context", "prefixes"),
    [
        ("evaluation", ("ratchet.measurement", "ratchet.optimization")),
        (
            "models",
            (
                "ratchet.backends.xpu",
                "ratchet.backends.cuda",
                "ratchet.backends.hip",
            ),
        ),
        (
            "measurement",
            ("ratchet.backends.xpu", "ratchet.backends.cuda", "ratchet.backends.hip"),
        ),
        ("experiments", ("torch.cuda", "torch.xpu")),
        ("reporting", ("ratchet.measurement",)),
        ("optimization", ("ratchet.oracle",)),
    ],
)
def test_forbidden_context_imports_are_absent(context: str, prefixes: tuple[str, ...]):
    _assert_no_import_prefix(context, prefixes)


@pytest.mark.parametrize("context", sorted(PUBLIC_CONTEXTS))
def test_cross_context_imports_target_only_public_entry_points(context: str):
    for path in _context_files(context):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _assert_cross_context_imports_are_public(context, _imported_modules(tree))


def test_nested_cross_context_import_is_rejected():
    tree = ast.parse("import ratchet.backends.xpu")

    with pytest.raises(AssertionError, match="ratchet.backends.xpu"):
        _assert_cross_context_imports_are_public("models", _imported_modules(tree))


@pytest.mark.parametrize(
    "context",
    [
        "evaluation",
        "models",
        "backends",
        "measurement",
        "experiments",
        "dispatch",
        "optimization",
        "reporting",
    ],
)
def test_public_contract_entry_points_are_torch_free(context: str):
    entry_point = ROOT / "ratchet" / context / "__init__.py"
    imports = _imports(entry_point)
    assert not {module for module in imports if module.startswith(("torch", "triton"))}
