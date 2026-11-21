"""Measures how many of 24 gross-positive signal candidates flip negative
once a one-bar implementation lag and an asset-class half-spread cost are
charged. Run from the repo root: python scripts/run_signal_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ptf.signal_backtest import run_all_candidates, summarize  # noqa: E402


def main() -> None:
    results = run_all_candidates()
    summary = summarize(results)

    print(f"{'name':28s} {'asset_class':9s} {'window':7s} {'gross':>14s} {'net':>14s} {'flipped':>8s}")
    for r in results:
        print(
            f"{r.spec.name:28s} {r.spec.asset_class:9s} {r.spec.window_events:7d} "
            f"{r.gross_total_return:14.6f} {r.net_total_return:14.6f} {str(r.flipped):>8s}"
        )

    print(f"\ncandidates: {summary['n_candidates']}")
    print(f"gross-positive candidates: {summary['n_gross_positive']}")
    print(f"of those, flipped negative after lag + cost: {summary['n_flipped_negative']}")
    print(f"flipped: {summary['flipped_names']}")
    print(f"\nRESULT: {summary['n_flipped_negative']} of {summary['n_gross_positive']} "
          f"gross-positive candidates turned negative  (claimed 9 of 24)")


if __name__ == "__main__":
    main()
