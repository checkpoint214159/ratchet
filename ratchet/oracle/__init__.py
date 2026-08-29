"""ZONE A -- the oracle. Immutable during optimization.

Nothing in this package may be edited as part of an optimization step. If a kernel only
passes after you change something here, the kernel is wrong.

scripts/check-oracle.sh enforces this with a checksum manifest. It is detection, not
prevention -- a determined process could rewrite both -- but it converts a silent
catastrophe into a loud one, which is the realistic goal.
"""
from .correctness import (  # noqa: F401
                     ABS_TOL,
                     REL_TOL,
                     CorrectnessResult,
                     DeterministicContext,
                     check_all,
                     check_determinism,
                     check_nonfinite,
                     check_tolerance,
)
from .device import DeviceProfile, calibrate, smem_at_least  # noqa: F401
from .inputs import (  # noqa: F401
                     BENCHMARK_SHAPES,
                     CORRECTNESS_SHAPES,
                     DISTRIBUTIONS,
                     Shape,
                     correctness_suite,
                     generate,
                     iter_correctness_cases,
)
from .reference import (  # noqa: F401
                     baseline_family,
                     best_baseline,
                     reference_fp32,
                     reference_fp64,
)
from .timing import (  # noqa: F401
                     L2Flusher,
                     MethodDescriptor,
                     TimingStats,
                     annotate_launch_domination,
                     cross_check,
                     get_timer,
)
