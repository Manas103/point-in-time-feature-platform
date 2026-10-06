"""Runs the 40 seeded lookahead-leak scenarios and the 40 clean
(false-positive) scenarios and prints the full accounting. Same logic as
tests/test_leak_check.py, kept separate so there is a human-readable raw
transcript for docs/leak_check_output.txt.
"""
import sys

sys.path.insert(0, ".")
from ptf.asof import asof_join, naive_asof_join_buggy  # noqa: E402
from ptf.batch_pandas import compute_batch_feature  # noqa: E402
from ptf.leak_check import find_lookahead_violations  # noqa: E402
from ptf.leak_scenarios import LEAK_SPEC, build_scenarios  # noqa: E402


def main():
    feature_df, adversarial, clean = build_scenarios()
    computed = compute_batch_feature(feature_df, LEAK_SPEC)

    correct_adv = asof_join(computed, adversarial, LEAK_SPEC)
    naive_adv = naive_asof_join_buggy(computed, adversarial, LEAK_SPEC)
    correct_adv_violations = find_lookahead_violations(correct_adv, LEAK_SPEC)
    naive_adv_violations = find_lookahead_violations(naive_adv, LEAK_SPEC)

    correct_clean = asof_join(computed, clean, LEAK_SPEC)
    naive_clean = naive_asof_join_buggy(computed, clean, LEAK_SPEC)
    correct_clean_violations = find_lookahead_violations(correct_clean, LEAK_SPEC)
    naive_clean_violations = find_lookahead_violations(naive_clean, LEAK_SPEC)

    print("=== 40 seeded lookahead-leak scenarios ===")
    print(f"correct backward-only join:  {len(correct_adv_violations)} / 40 leaked (want 0)")
    print(f"naive nearest-match join:    {len(naive_adv_violations)} / 40 leaked "
          f"(must be >0, or the scenarios are vacuous)")
    caught = 40 - len(correct_adv_violations)
    print(f"-> correct join caught/prevented {caught} of 40 seeded lookahead leaks")

    print()
    print("=== 40 clean (unambiguous, non-adversarial) scenarios ===")
    print(f"correct backward-only join: {len(correct_clean_violations)} / 40 falsely flagged (want 0)")
    print(f"naive nearest-match join:   {len(naive_clean_violations)} / 40 falsely flagged (want 0)")

    ok = (caught == 40) and (len(naive_adv_violations) == 40) and (len(correct_clean_violations) == 0)
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
