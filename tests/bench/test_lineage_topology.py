"""The tree must actually be a tree.

`bench/README.md` states the premise: git branches ARE the evolutionary tree, lineage is
ancestry, CMP is forward reachability. Finding 28 measured that the repository did not
satisfy it -- every candidate branch was cut from `ben`'s tip, and because every candidate
is merged INTO `ben`, each new branch inherited every earlier candidate. Each candidate had
exactly `generation - 1` git ancestors: a perfectly linear chain, which is the L1
degeneracy the method exists to escape.

These tests pin the true lineage (the registry), and enforce the branching discipline for
every candidate created after the flaw was found.
"""
import subprocess

import pytest

from bench.candidates import REGISTRY
from bench.ledger import BenchLedger, candidate_descendants, declared_lineage

# Candidates created before finding 28. Their git topology is wrong and cannot be fixed
# without rewriting history, which the contract forbids (never rebase, squash or amend --
# it silently reparents the tree). They are scored by declared lineage like everything
# else; only the git-agreement rule is waived.
PRE_FINDING_28 = {n for n, s in REGISTRY.items() if s.generation <= 18}


def _sha_of(candidate: str) -> str | None:
    for r in BenchLedger().clean_rows():
        if r.get("candidate") == candidate:
            return r["commit_sha"]
    return None


def test_every_declared_parent_exists_in_the_registry():
    lineage = declared_lineage()
    for child, parent in lineage.items():
        assert parent is None or parent in lineage, f"{child} claims unknown parent {parent}"


def test_the_declared_lineage_is_acyclic():
    lineage = declared_lineage()
    for name in lineage:
        seen, cur = set(), name
        while cur is not None:
            assert cur not in seen, f"cycle through {cur}"
            seen.add(cur)
            cur = lineage.get(cur)


def test_declared_lineage_actually_branches():
    """The whole point. If every node has at most one child, CMP is measuring age."""
    children: dict[str, int] = {}
    for _child, parent in declared_lineage().items():
        if parent:
            children[parent] = children.get(parent, 0) + 1
    forks = {p: n for p, n in children.items() if n > 1}
    assert forks, "the declared lineage is a chain, not a tree -- CMP would be degenerate"


def test_descendants_of_a_leaf_is_just_itself():
    leaves = [n for n in REGISTRY
              if not any(s.parent == n for s in REGISTRY.values())]
    assert leaves
    for leaf in leaves:
        assert candidate_descendants(leaf) == {leaf}


def test_a_forks_descendants_include_both_siblings():
    """v9a and v9b are the g9 fork off v8_padfast. v8's clade must contain both, which is
    exactly the property that makes it a good ancestor rather than a mediocre node."""
    d = candidate_descendants("v8_padfast")
    assert {"v9a_compiled_core", "v9b_reduce_overhead"} <= d


@pytest.mark.parametrize(
    "name", sorted(n for n, s in REGISTRY.items() if s.generation > 18))
def test_new_candidates_branch_from_their_declared_parent(name):
    """THE DISCIPLINE, enforced from generation 19 onward.

    A candidate must be cut from its declared parent's commit, not from `ben`'s tip. The
    set of candidates that are its git ancestors must equal the set of its declared
    ancestors -- no more. Inheriting an unrelated sibling means the branch was taken from
    the integration trunk, and CMP silently degenerates again.

    Harness and tooling changes are NOT candidates, so merging them in is fine.
    """
    sha = _sha_of(name)
    if sha is None:
        pytest.skip(f"{name} has no measured rows yet")

    declared, cur = set(), REGISTRY[name].parent
    while cur is not None:
        declared.add(cur)
        cur = REGISTRY[cur].parent if cur in REGISTRY else None

    actual = set()
    for other in REGISTRY:
        if other == name:
            continue
        osha = _sha_of(other)
        if osha and subprocess.run(["git", "merge-base", "--is-ancestor", osha, sha],
                                   capture_output=True).returncode == 0:
            actual.add(other)

    extra = actual - declared - PRE_FINDING_28
    assert not extra, (
        f"{name} git-descends from {sorted(extra)}, which are not its declared "
        f"ancestors. It was branched from the integration trunk instead of from "
        f"{REGISTRY[name].parent}. See finding 28.")
