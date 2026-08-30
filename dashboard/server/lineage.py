"""Emit the DECLARED lineage as JSON, for the dashboard's evolution tree.

THE POINT (docs/findings/28-the-tree-was-a-chain.md). `bench/README.md` says "git
branches are the evolutionary tree". That was the intent and it is not what the
repository contains: every candidate branch was cut from `ben`'s tip, and every
candidate is merged INTO `ben`, so each new branch inherited every earlier candidate.
Measured, each candidate has exactly `generation - 1` git ancestors -- a perfectly
linear chain. The spurs in `git log --graph` are decorative.

The real tree is `CandidateSpec.parent` in `bench/candidates/__init__.py`, which is
what `clade_stats_by_candidate` / `sample_candidate` have scored over since finding 28.
It genuinely branches. So the dashboard draws THIS, not git ancestry.

Read-only: imports the registry, reads two test files as text, writes nothing.
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TESTS = REPO / "tests" / "bench"


def _literal_assignments(path: pathlib.Path, want: set[str]) -> dict[str, object]:
    """Pull literal assignments out of a source file WITHOUT importing it.

    `tests/bench/test_lineage_invariants.py` imports torch at module scope; the
    dashboard must never pay for that (or need a GPU) to draw a tree. The two facts
    wanted here are plain set literals, so `ast` reads them directly. A name that is
    not a literal is skipped rather than guessed at.
    """
    out: dict[str, object] = {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except OSError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in want:
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass
    return out


_GEN_REF = re.compile(r"\bg(\d+)\b")


def _recombinations(registry) -> dict[str, list[str]]:
    """Secondary contributors, derived from the summary text.

    `CandidateSpec` has ONE parent field, so a recombination -- v17 merges the g16 FFN
    megakernel into the g13 frontier -- can only be recorded in prose today. Rather
    than drop the edge (merges expressing recombination are a documented feature of
    this design, not noise), it is recovered from summaries that announce themselves
    with the word RECOMBINATION: every other `gNN` they cite, mapped to the
    candidate(s) of that generation, minus the declared parent's own generation.

    This is a READER of an existing convention, deliberately conservative: a summary
    that does not say RECOMBINATION contributes no edge, and a generation reference
    that matches no candidate is dropped. If `CandidateSpec` ever grows a real
    `contributors` field, prefer it and delete this.
    """
    by_gen: dict[int, list[str]] = {}
    for spec in registry.values():
        by_gen.setdefault(spec.generation, []).append(spec.name)

    out: dict[str, list[str]] = {}
    for name, spec in registry.items():
        summary = spec.summary or ""
        if "RECOMBINATION" not in summary:
            continue
        parent_gen = registry[spec.parent].generation if spec.parent in registry else None
        secondary: list[str] = []
        for hit in _GEN_REF.findall(summary):
            gen = int(hit)
            if gen == spec.generation or gen == parent_gen:
                continue
            for other in by_gen.get(gen, ()):
                if other != name and other != spec.parent and other not in secondary:
                    secondary.append(other)
        if secondary:
            out[name] = secondary
    return out


def main() -> None:
    from bench.candidates import REGISTRY

    unsafe = _literal_assignments(
        TESTS / "test_lineage_invariants.py", {"known_unsafe"}
    ).get("known_unsafe") or ()
    violations = _literal_assignments(
        TESTS / "test_lineage_topology.py", {"KNOWN_VIOLATIONS"}
    ).get("KNOWN_VIOLATIONS") or ()
    unsafe = sorted(unsafe)
    violations = sorted(violations)

    recomb = _recombinations(REGISTRY)
    children: dict[str, list[str]] = {}
    for name, spec in REGISTRY.items():
        if spec.parent:
            children.setdefault(spec.parent, []).append(name)

    nodes = []
    for name, spec in REGISTRY.items():
        nodes.append({
            "name": name,
            "generation": spec.generation,
            "parent": spec.parent,
            "summary": spec.summary,
            # A parent named but absent from the registry would be a dangling edge; the
            # tree treats such a node as a root and says so rather than dropping it.
            "parent_known": bool(spec.parent) and spec.parent in REGISTRY,
            "children": sorted(children.get(name, ())),
            "recombines": recomb.get(name, []),
            "known_unsafe": name in unsafe,
            "topology_violation": name in violations,
        })
    nodes.sort(key=lambda n: (n["generation"], n["name"]))

    print(json.dumps({
        "nodes": nodes,
        # Secondary contribution edges, kept separate from the tree edges so the layout
        # can draw them differently. They are real; they are not tree edges.
        "recombination_edges": [
            {"from": src, "to": name}
            for name, srcs in sorted(recomb.items()) for src in srcs
        ],
        "known_unsafe": unsafe,
        "topology_violations": violations,
    }))


if __name__ == "__main__":
    main()
