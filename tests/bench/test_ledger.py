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


def test_scoreboard_maps_score_to_commit(tmp_path):
    led = BenchLedger(tmp_path / "r.jsonl")
    for cfg in range(1, 15):
        led.append(_row("c" * 40, cfg, 2.0))
    board = scoreboard(led)
    assert board[0]["commit_sha"] == "c" * 40
    assert board[0]["configs_passed"] == 14
    assert board[0]["weighted_score"] == 2.0


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
