#!/usr/bin/env python
"""Run one ensemble of a configured job. The entry point a SLURM task calls.

    python scripts/run_ensemble.py --config run.toml --ensemble 3 --out runs/123

Writes ``ensemble_003.npz`` under ``--out``, containing the **full** trace with
burn-in included; how much to discard is decided at merge time.
"""

# sys.path is set before the package import below: the editable install is
# unreliable here, because macOS flags .pth files under iCloud-synced folders
# hidden and CPython silently skips those. See pyproject.toml.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nuclear_spin_recovery import RunConfig, run_ensemble


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ensemble", required=True, type=int,
                        help="ensemble index, e.g. $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: the config's)")
    args = parser.parse_args(argv)

    config = RunConfig.from_toml(args.config)
    out = args.out if args.out is not None else Path(config.output["dir"])
    # The resolved configuration goes next to the results, written by every
    # task. Identical bytes from each, so a race is harmless and a missing file
    # means no task got that far.
    config.write_resolved(out / "resolved.json")
    trace = run_ensemble(config, args.ensemble, out_dir=out)
    print(f"ensemble {args.ensemble}: {len(trace)} steps -> "
          f"{out / f'ensemble_{args.ensemble:03d}.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
