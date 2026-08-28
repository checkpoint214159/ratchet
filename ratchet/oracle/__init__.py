"""ZONE A -- the oracle. Immutable during optimization.

Nothing in this package may be edited as part of an optimization step. If a kernel only
passes after you change something here, the kernel is wrong.

scripts/check-oracle.sh enforces this with a checksum manifest. It is detection, not
prevention -- a determined process could rewrite both -- but it converts a silent
catastrophe into a loud one, which is the realistic goal.
"""
from .device import DeviceProfile, calibrate, smem_at_least          # noqa: F401
from .inputs import (Shape, CORRECTNESS_SHAPES, BENCHMARK_SHAPES,    # noqa: F401
                     DISTRIBUTIONS, generate, correctness_suite,
                     iter_correctness_cases)
from .correctness import (REL_TOL, ABS_TOL, CorrectnessResult,       # noqa: F401
                          DeterministicContext, check_all, check_tolerance,
                          check_nonfinite, check_determinism)
from .reference import (reference_fp32, reference_fp64,              # noqa: F401
                        baseline_family, best_baseline)
from .timing import (TimingStats, MethodDescriptor, get_timer,       # noqa: F401
                     cross_check, annotate_launch_domination, L2Flusher)
