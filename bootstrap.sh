#!/usr/bin/env bash
# Ratchet bootstrap. Run in WSL2, not PowerShell.
#
# VENDORED into the repo 2026-08-28 for reproduction (M13). One patch vs the
# original: --test-runner pytest -> pytest-testmon ("pytest" is not a valid value
# in the pinned Beryl revision and fails STEP 5). pip install pytest-testmon first.
#
# ---------------------------------------------------------------------------------------
# READ THIS BEFORE RUNNING
#
# Beryl (github.com/Praneeth-Suresh/Beryl) is used here as the agent control plane. It is
# a genuinely well-engineered project and it works -- I installed and ran it -- but be
# clear-eyed about what it is:
#
#   * v0.1.0, one release, 0 stars, 0 forks, single author, no external coverage anywhere.
#   * It installs a CONTROL PLANE, not a project. Zero Python files, no pyproject.toml,
#     no source directories. It gives you routing, skills, a plan-ratification gate and a
#     deterministic check script. You still write the code.
#   * The signed-release metadata EXPIRES 2026-09-20 and the bootstrap aborts on expired
#     metadata. That is why STEP 2 below uses a local checkout rather than the remote
#     signed path -- the local path sidesteps the expiry entirely.
#   * The `full` profile drops the author's OWN driver tasks into your repo (five briefs
#     referencing Beryl's GitHub issues). STEP 5 deletes them. Its config.example.env
#     also carries leftovers from an unrelated project.
#
# The valuable part is `.beryl/agent/` -- the routing table, the twelve skill files, and
# the plan-before-edit gate. That is just Markdown you can read, diff and prune in an
# afternoon. If any step below misbehaves, the FALLBACK is: skip Beryl entirely, keep
# CLAUDE.md and the docs/ + specs/ tree from this package, and lose nothing important.
#
# Git hooks are deliberately OFF at first. Beryl's test-immutability manifest blocks every
# commit that touches a test file until you rerun update-test-manifest.sh, which is real
# friction on a harness whose reference tests change constantly. Turn it on at STEP 8,
# scoped to the golden correctness tests only.
# ---------------------------------------------------------------------------------------

set -euo pipefail

PROJECT="${PROJECT:-$HOME/ratchet}"
BERYL="${BERYL:-$HOME/.beryl-src}"
HANDOFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------------------
say "STEP 0  preflight"
# ---------------------------------------------------------------------------------------
command -v git >/dev/null || { echo "git required"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }

python3 - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("torch not importable -- install it before continuing")
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False. Everything here needs a real GPU.")
p = torch.cuda.get_device_properties(0)
print(f"  {p.name}  sm_{p.major}{p.minor}  {p.multi_processor_count} SMs")
print(f"  smem/block optin {p.shared_memory_per_block_optin/1024:.0f} KB   "
      f"L2 {p.L2_cache_size/1024/1024:.0f} MB")
try:
    import triton; print(f"  triton {triton.__version__}")
except ImportError:
    sys.exit("triton not importable -- install it before continuing")
print(f"  torch {torch.__version__}")
PY
echo "  ^ paste this into docs/00-mission.md under 'Target hardware' before going further"

# ---------------------------------------------------------------------------------------
say "STEP 1  repo shell (Beryl will NOT create this for you)"
# ---------------------------------------------------------------------------------------
mkdir -p "$PROJECT" && cd "$PROJECT"
if [ ! -d .git ]; then
  git init -q
  printf '# ratchet\n\nAgentic GPU kernel optimization harness.\n' > README.md
  cat > .gitignore <<'EOF'
__pycache__/
.venv/
*.egg-info/
.pytest_cache/
.testmondata
ledger/artifacts/
ledger/logs/
ledger/state/
EOF
  mkdir -p ratchet/{oracle,kernels,dispatch,search,critic,report} tests scripts ledger prompts
  git add -A && git commit -qm "chore: empty scaffold"
fi

# ---------------------------------------------------------------------------------------
say "STEP 2  copy the handoff package in"
# ---------------------------------------------------------------------------------------
cp -r "$HANDOFF_DIR/docs" "$HANDOFF_DIR/specs" "$PROJECT/"
cp "$HANDOFF_DIR/CLAUDE.md" "$HANDOFF_DIR/HANDOFF.md" "$PROJECT/"
cp -r "$HANDOFF_DIR/seed/ratchet/." "$PROJECT/ratchet/"
cp -r "$HANDOFF_DIR/seed/prompts/." "$PROJECT/prompts/" 2>/dev/null || true
echo "  docs/ specs/ CLAUDE.md HANDOFF.md and the seed oracle are in place"

# ---------------------------------------------------------------------------------------
say "STEP 3  Beryl, from a local checkout (avoids the 2026-09-20 metadata expiry)"
# ---------------------------------------------------------------------------------------
if [ ! -d "$BERYL/.git" ]; then
  git clone -q https://github.com/Praneeth-Suresh/Beryl.git "$BERYL"
fi
BERYL_REF="$(git -C "$BERYL" rev-parse HEAD)"
echo "  pinned to $BERYL_REF  -- record this in docs/00-mission.md"

# ---------------------------------------------------------------------------------------
say "STEP 4  DRY RUN first. Never skip this on a v0.1.0 tool."
# ---------------------------------------------------------------------------------------
sh "$BERYL/install.sh" --source-dir "$BERYL" --ref "$BERYL_REF" \
   --dry-run --profile full --target "$PROJECT" || {
     echo
     echo "  Beryl dry-run failed. FALLBACK: skip steps 4-8 entirely."
     echo "  You keep CLAUDE.md, docs/ and specs/, which is the part that matters."
     exit 0
   }

read -r -p "  Proceed with the real install? [y/N] " ok
[ "${ok:-n}" = "y" ] || { echo "  stopping; the fallback above is fine"; exit 0; }

# ---------------------------------------------------------------------------------------
say "STEP 5  install (git hooks OFF for now -- see the header)"
# ---------------------------------------------------------------------------------------
"$BERYL/.beryl/scripts/setup-project.sh" --non-interactive \
  --profile full --stack python --test-runner pytest-testmon \
  --root-conflict skip \
  "$PROJECT"

# Beryl's `full` profile ships the AUTHOR'S OWN driver tasks. Remove them.
rm -f "$PROJECT/.beryl/driver/tasks/"*.md
if [ -f "$PROJECT/.beryl/driver/config.example.env" ]; then
  cp "$PROJECT/.beryl/driver/config.example.env" "$PROJECT/.beryl/driver/config.env"
  echo "  edit .beryl/driver/config.env:"
  echo "    DRIVER_AGENT=\"claude\""
  echo "    VERIFY_STACK_MODE=\"never\"      # 'auto' assumes a web stack; meaningless here"
  echo "    WORK_BRANCH=\"feat/tier0\"       # the default is Beryl's own branch name"
  echo "    DRIVER_UNATTENDED_OK stays \"false\" until you have watched a full run"
fi

# ---------------------------------------------------------------------------------------
say "STEP 6  teach Beryl's gate about GPU paths"
# ---------------------------------------------------------------------------------------
cat <<'EOF'
  In .beryl/agent/affected-tests.conf, add to RELATED_CHANGE_GLOBS:
      "ratchet/**"  "tests/**"
  (the defaults cover src/ lib/ app/ packages/ and will miss everything here)

  In .beryl/agent/test-manifest.conf, narrow INCLUDE_GLOBS to the GOLDEN correctness
  tests only. If it tracks every test file, autotuning churn trips the gate constantly
  and you will end up disabling it in frustration, which loses the one thing it is for:
  detecting an agent silently weakening a correctness test.
EOF

# ---------------------------------------------------------------------------------------
say "STEP 7  the oracle checksum manifest -- this is OUR gate, not Beryl's"
# ---------------------------------------------------------------------------------------
cat > "$PROJECT/scripts/check-oracle.sh" <<'EOF'
#!/usr/bin/env bash
# Zone A integrity. Detection, not prevention -- but it turns a silent catastrophe into
# a loud one, which is the realistic goal.
set -euo pipefail
cd "$(dirname "$0")/.."
MANIFEST="ratchet/oracle/.manifest.sha256"

fail=0

if [ ! -f "$MANIFEST" ]; then
  echo "no manifest; creating one (do this deliberately, on a clean tree)"
  find ratchet/oracle -name '*.py' -print0 | sort -z | xargs -0 sha256sum > "$MANIFEST"
  echo "created $MANIFEST"
else
  if ! sha256sum -c --quiet "$MANIFEST" 2>/dev/null; then
    echo "ORACLE MODIFIED. Zone A is immutable during optimization."
    echo "If a kernel only passes after this change, the kernel is wrong."
    echo "If the change is legitimate: separate branch, human approval, then"
    echo "  find ratchet/oracle -name '*.py' -print0 | sort -z | xargs -0 sha256sum > $MANIFEST"
    fail=1
  else
    echo "oracle: OK"
  fi
fi

# The ledger must never be opened for writing in anything but append mode.
if grep -rnE '(measurements\.jsonl["'"'"']?\s*,\s*["'"'"'](w|r\+))|open\([^)]*measurements[^)]*["'"'"']w' \
     --include='*.py' ratchet/ 2>/dev/null; then
  echo "LEDGER OPENED FOR WRITING. Zone C is append-only."
  fail=1
else
  echo "ledger append-only: OK"
fi

# Tolerances must not be reassigned anywhere outside their definition.
if grep -rnE '^\s*(REL_TOL|ABS_TOL)\s*=' --include='*.py' ratchet/ \
     | grep -v 'ratchet/oracle/correctness.py'; then
  echo "TOLERANCE REDEFINED OUTSIDE THE ORACLE. These are locked constants."
  fail=1
else
  echo "tolerances locked: OK"
fi

exit $fail
EOF
chmod +x "$PROJECT/scripts/check-oracle.sh"
"$PROJECT/scripts/check-oracle.sh" || true

# ---------------------------------------------------------------------------------------
say "STEP 8  verify green, then commit"
# ---------------------------------------------------------------------------------------
cd "$PROJECT"
[ -x ./.beryl/scripts/check.sh ] && ./.beryl/scripts/check.sh || true
git add -A && git commit -qm "chore: ratchet handoff + Beryl control plane" || true

cat <<'EOF'

  Done. Next:

    python -m ratchet.oracle.device        # M1 -- calibrate, then paste into 00-mission.md

  Then open Claude Code in this directory and send:

    Read HANDOFF.md and CLAUDE.md, then docs/00-mission.md and docs/04-failure-modes.md.
    Fill in the Target hardware table from the calibration output. Then work milestone M1
    through M3 in docs/02-milestones.md. Do not start the search loop until the Tier 0
    acceptance gate is green -- all five conditions, including that timing the reference
    against itself gives 1.00x.

  Later, once the correctness test surface is stable, enable Beryl's pre-commit hook:

    sh "$BERYL/install.sh" --source-dir "$BERYL" --ref "$BERYL_REF" \
       --update --enable-githooks --target "$PROJECT"
EOF
