"""Git-as-evolutionary-tree: pin the lineage and provenance rules."""
import json

from bench.ledger import (BenchLedger, clade_stats, scoreboard, descendants,
                          provenance, head_sha)


def _row(sha, cfg, speedup, passed=True, dirty=False, status="ok"):
    return {"ts": "2026-08-29T00:00:00Z", "commit_sha": sha, "branch": "t",
            "dirty": dirty, "candidate": "c", "config_id": cfg, "status": status,
            "correctness": {"passed": passed}, "timing": {"speedup": speedup}}


def test_append_is_the_only_write_path(tmp_path):
    led = BenchLedger(tmp_path / "r.jsonl")
    assert not hasattr(led, "update") and not hasattr(led, "delete")
    led.append(_row("a" * 40, 1, 1.8))
    led.append(_row("a" * 40, 2, 1.2))
    assert len(list(led.rows())) == 2


def test_append_rejects_a_row_without_provenance(tmp_path):
    led = BenchLedger(tmp_path / "r.jsonl")
    for missing in ("commit_sha", "config_id", "status"):
        row = _row("a" * 40, 1, 1.8)
        del row[missing]
        try:
            led.append(row)
        except ValueError:
            continue
        raise AssertionError(f"append accepted a row missing {missing!r}")


def test_truncated_final_line_does_not_break_reading(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(_row("a" * 40, 1, 1.8)) + "\n" + '{"partial": ')
    assert len(list(BenchLedger(p).rows())) == 1


def test_dirty_rows_are_recorded_but_excluded_from_clade_stats(tmp_path):
    led = BenchLedger(tmp_path / "r.jsonl")
    led.append(_row("b" * 40, 1, 2.0, dirty=True))
    assert len(list(led.rows())) == 1, "dirty rows are still evidence"
    assert list(led.clean_rows()) == [], "but a dirty sha is a false provenance claim"


def _compiled_row(cfg, ms):
    """A baseline_compiled row -- the reference honest scores are quoted against."""
    return {"ts": "2026-08-29T00:00:00Z", "commit_sha": "0" * 40, "branch": "t",
            "dirty": False, "candidate": "baseline_compiled", "config_id": cfg,
            "status": "ok", "correctness": {"passed": True},
            "timing": {"candidate_ms": ms, "speedup": 1.0}}


def test_scoreboard_maps_score_to_commit(tmp_path):
    led = BenchLedger(tmp_path / "r.jsonl")
    for cfg in range(1, 15):
        led.append(_compiled_row(cfg, 4.0))          # compiled reference: 4ms
        row = _row("c" * 40, cfg, 2.0)
        row["timing"]["candidate_ms"] = 2.0          # candidate: 2ms -> 2x vs compiled
        led.append(row)
    board = [e for e in scoreboard(led) if e["candidate"] != "baseline_compiled"][0]
    assert board["commit_sha"] == "c" * 40
    assert board["configs_passed"] == 14
    assert board["weighted_score"] == 2.0, "scored against the COMPILED baseline"


def test_score_is_zero_without_a_compiled_reference(tmp_path):
    # Refusing to score is correct: quoting against eager is what finding 12 showed
    # produces a saturated, inverted ranking. Better no number than a misleading one.
    led = BenchLedger(tmp_path / "r.jsonl")
    for cfg in range(1, 15):
        led.append(_row("d" * 40, cfg, 2.0))
    assert scoreboard(led)[0]["weighted_score"] == 0.0


def test_failing_rows_never_count_as_clade_successes(tmp_path):
    # A round that fixed a compile error without moving the timing is a failure to
    # improve, not progress -- otherwise the loop drifts toward safe, slow, correct code.
    led = BenchLedger(tmp_path / "r.jsonl")
    led.append(_row("d" * 40, 1, 0.9))            # correct but slower than baseline
    led.append(_row("d" * 40, 2, 1.5, passed=False))
    stats = clade_stats(led, repo=None)
    assert stats.get("d" * 40, (0, 0))[0] == 0


def test_descendants_includes_self_and_reaches_the_root():
    import subprocess
    root = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"],
                          capture_output=True, text=True).stdout.split()[0]
    head = head_sha()
    assert head in descendants(head)
    assert head in descendants(root), "HEAD must be reachable forward from the root"


def test_provenance_reports_the_real_head():
    p = provenance()
    assert p["commit_sha"] == head_sha() and len(p["commit_sha"]) == 40
    assert isinstance(p["dirty"], bool)


def test_baseline_rows_are_not_clade_failures(tmp_path):
    # A baseline row has speedup 1.0 by definition. Counting it as a candidate failure
    # would book one guaranteed loss per config and drag every posterior toward zero.
    led = BenchLedger(tmp_path / "r.jsonl")
    row = _row("e" * 40, 1, 1.0)
    row["candidate"] = "baseline"
    led.append(row)
    led.append(_row("e" * 40, 1, 2.0))          # a real candidate win
    s, f = clade_stats(led, repo=None)["e" * 40]
    assert (s, f) == (1, 0)


def test_writing_the_ledger_does_not_dirty_the_run():
    # Self-contamination bug: the ledger is a tracked file, so recording a measurement
    # dirtied the tree and stamped every LATER run dirty, excluding it from its own
    # clade statistics. Appended data is not changed source.
    import subprocess
    from bench.ledger import is_dirty, DEFAULT_PATH
    porcelain = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout
    if DEFAULT_PATH.split("/")[-1] in porcelain and len(porcelain.strip().splitlines()) == 1:
        assert not is_dirty(), "a modified ledger alone must not mark the tree dirty"


def test_scoreboard_separates_padding_conditions(tmp_path):
    # v8 was measured at padding 0.0 and 0.5. Pooling them collapses two different
    # measurement conditions onto one key, and last-write-wins then reports whichever
    # happened to run last as if it were the candidate's score.
    led = BenchLedger(tmp_path / "r.jsonl")
    # below the 3x clip, or both would saturate and the comparison says nothing
    for cfg in range(1, 15):
        led.append(_compiled_row(cfg, 5.0))
    for pad, ms in ((0.0, 2.0), (0.5, 2.8)):        # 2.5x and ~1.79x vs compiled
        for cfg in range(1, 15):
            row = _row("f" * 40, cfg, 5.0 / ms)
            row["padding_ratio"] = pad
            row["timing"]["candidate_ms"] = ms
            led.append(row)
    board = [e for e in scoreboard(led) if e["candidate"] == "c"]
    assert len(board) == 2, "each padding condition is its own scoreboard entry"
    assert {round(e["padding_ratio"], 2) for e in board} == {0.0, 0.5}
    assert board[0]["weighted_score"] > board[1]["weighted_score"]


def test_scoreboard_counts_configs_not_rows(tmp_path):
    # A parameter sweep revisits the same config many times. Counting rows reported
    # "56 configs measured" on a 14-config matrix -- a number that cannot exist.
    led = BenchLedger(tmp_path / "r.jsonl")
    for _ in range(8):
        for cfg in (1, 2):
            led.append(_row("a" * 40, cfg, 2.0))
    e = scoreboard(led)[0]
    assert e["configs_measured"] == 2, "distinct configs, not rows"
    assert e["rows"] == 16
    assert e["is_sweep"] is True
