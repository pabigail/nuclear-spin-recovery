#!/usr/bin/env python
"""Pool the ensembles a job array produced.

    python scripts/merge_ensembles.py --config run.toml --out runs/123

Refuses to pool fewer traces than the configuration declares unless
``--allow-partial`` is given: a silently partial merge is how a failed array
task becomes a published posterior.
"""

# sys.path is set before the package import below: the editable install is
# unreliable here, because macOS flags .pth files under iCloud-synced folders
# hidden and CPython silently skips those. See pyproject.toml.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nuclear_spin_recovery import RunConfig, merge_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true",
                        help="pool an incomplete array anyway")
    args = parser.parse_args(argv)

    config = RunConfig.from_toml(args.config)
    result = merge_run(config, args.out, allow_partial=args.allow_partial)
    agreement = result.agreement()

    print(f"pooled {len(result.traces)} ensembles, {len(result.pooled)} draws")
    print(f"init policy : {result.init_name}")
    print(f"modal k     : "
          f"{[int(np.bincount(np.asarray(t.k)).argmax()) for t in result.traces]}")
    print(f"k spread    : {agreement.k_mode_spread}")
    for name, value in agreement.rhat.items():
        shown = "nan (never moved)" if np.isnan(value) else f"{value:.3f}"
        print(f"R-hat {name:10s}: {shown}")
    print("\nR-hat is reported, never gated: on this problem it is structurally "
          "above 1\nand rises with chain length while the pooled mode stays "
          "correct. See\ndocs/phase-4-plan.md section 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
