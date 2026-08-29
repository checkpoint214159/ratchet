# 28 — The evolutionary tree was a chain, and I rebuilt the exact degeneracy I documented

**Date:** 2026-08-30. **Found by:** the user, looking at the dashboard and asking why the
graph showed one long lineage instead of branching paths.

## The premise, and what the repository actually contained

`bench/README.md` states it plainly: **git branches are the evolutionary tree**, a
candidate is a commit, lineage is ancestry, CMP is forward reachability. Measured:

    candidate                     git-ancestors that are candidates    declared parent
    v9a_compiled_core                            8                     v8_padfast
    v9b_reduce_overhead                          8                     v8_padfast
    v13_safe_capture                            12                     v12_graph_over_compile
    v15_lifted_veto                             14                     v9b_reduce_overhead
    v16_ffn_megakernel                          15                     v9b_reduce_overhead
    v17_dispatched_megakernel                   16                     v13_safe_capture
    v18_capture_insurance                       17                     v17

**Every candidate has exactly `generation - 1` git ancestors.** A perfectly linear chain.
No branching whatsoever, at any point in eighteen generations.

## The cause, which was my branching discipline

Each candidate branch was cut from `ben`'s tip, so it would inherit the latest harness --
the stage-1 screen, the GPU lock, the `__ROW__` correctness fix, the corrected `matrix.py`.
And every candidate is merged back INTO `ben`. So cutting from `ben` inherits every earlier
candidate, and the topology collapses no matter how the branches are named.

The branches look like a tree in `git log --graph`: short spurs off a trunk, each merging
back. They are decorative. `merge-base --is-ancestor` says they are a line.

## This is L1, rebuilt by the person who wrote L1

L1, recorded on day one:

> On a single branch every later commit is a git descendant of every earlier one, so a
> node's clade is just "everything committed after it" and the ranking measures **age, not
> productivity**. HGM's whole premise requires the subtree to be *chosen*, which requires
> branching. **Branch first**; the sampling becomes meaningful once siblings exist.

I then created eighteen generations of branches that satisfy the letter of "branch first"
and none of its content. Finding 21 later fixed the clade *success criterion* and measured
the age correlation down to -0.158 — which is why the degeneracy stayed hidden: the
criterion patch compensated for a topology that was still wrong.

## Impact: real, and so far harmless

CMP over the true (declared) lineage against CMP over git ancestry:

    candidate                declared-CMP        git-CMP
    v9b_reduce_overhead      12w/27l  0.317      24w/141l  0.150
    v8_padfast               38w/105l 0.269      35w/172l  0.172
    v9a_compiled_core        15w/76l  0.172      23w/128l  0.157

    top-3 by declared lineage:  v9b, v8, v9a
    top-3 by git ancestry:      v8, v9a, v9b

**The same three nodes, reordered.** Git pools every downstream candidate into every
node, diluting all of them toward a common rate; the declared lineage keeps each clade to
what actually descends from it. No expansion was sent to the wrong place — but that is
luck, not design, and the dilution grows with every generation.

## The fix

1. **CMP now reads the registry**, via `clade_stats_by_candidate` / `sample_candidate`.
   The registry's `parent` field is the true record: v15 and v16 are siblings off v9b,
   v17 recombines v16 into v13. Git ancestry cannot express that here because the topology
   was built wrong, and history may not be rewritten to repair it (never rebase, squash or
   amend a candidate branch -- it silently reparents the tree).

2. **The branching discipline, enforced from generation 19** by
   `tests/bench/test_lineage_topology.py`:
   - a candidate is cut from **its declared parent's commit**, never from `ben`;
   - harness and tooling changes are not candidates, so merging them in is fine;
   - `ben` is a downstream **integration** branch and is never a branch point;
   - a merge between two candidate branches is a real recombination and is allowed.

   The test asserts that a new candidate's candidate-ancestors equal its declared
   ancestors exactly. Generations <= 18 are grandfathered, since their topology cannot be
   corrected without rewriting history.

## L40 — Writing down a lesson is not the same as building the thing it demands

L1 named this exact failure on day one, in this repository, and the fix ("branch first")
was implemented in a form that satisfied the words and none of the mechanism. The lesson
was cited in later findings as settled. What was missing was a **test**: nothing ever
asserted that the tree branched, so eighteen generations of linear history passed
unchallenged while the file explaining why that was fatal sat two directories away.

Every structural claim this project relies on should have an executable check. The ones
that do -- the oracle manifest, the append-only ledger, the tolerance lock -- have never
silently broken. The ones that live only in prose -- "git is the tree", "premises in
matrix.py are true" (finding 23) -- have both now been found false by someone looking
rather than by the system noticing.

Related: L36 (a test can pass because its subject was never built) and L38 (verify a check
can fail before trusting that it passed). This is the third member of that family in two
days, and the common shape is **an assurance nobody arranged to be capable of failing.**
