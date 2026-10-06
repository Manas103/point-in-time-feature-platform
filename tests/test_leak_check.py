from ptf.asof import asof_join, naive_asof_join_buggy
from ptf.batch_pandas import compute_batch_feature
from ptf.leak_check import find_lookahead_violations
from ptf.leak_scenarios import BASE_TIME_NS, LEAK_SPEC, SPACING_NS, build_scenarios


def _event_time(seq: int) -> int:
    return BASE_TIME_NS + seq * SPACING_NS


def test_correct_join_catches_all_40_seeded_leak_scenarios():
    feature_df, adversarial, _ = build_scenarios()
    computed = compute_batch_feature(feature_df, LEAK_SPEC)

    correct_result = asof_join(computed, adversarial, LEAK_SPEC)
    correct_violations = find_lookahead_violations(correct_result, LEAK_SPEC)
    assert len(correct_violations) == 0, "correct backward join must not leak on any seeded scenario"

    # Not just "no leak" -- actually matched the correct prior event, not a
    # vacuous non-match.
    expected_times = correct_result["expected_event_seq"].map(_event_time)
    assert (correct_result[LEAK_SPEC.order_key] == expected_times).all()

    # The scenarios must be genuinely leak-inducing, not vacuous: the naive
    # nearest-match join should leak on every single one of them.
    naive_result = naive_asof_join_buggy(computed, adversarial, LEAK_SPEC)
    naive_violations = find_lookahead_violations(naive_result, LEAK_SPEC)
    assert len(naive_violations) == 40, "seeded scenarios must actually defeat the naive join"


def test_zero_false_positives_on_clean_scenarios():
    feature_df, _, clean = build_scenarios()
    computed = compute_batch_feature(feature_df, LEAK_SPEC)

    correct_result = asof_join(computed, clean, LEAK_SPEC)
    violations = find_lookahead_violations(correct_result, LEAK_SPEC)
    assert len(violations) == 0

    # Unambiguous by construction: the naive join must also agree here,
    # which is what makes "clean" the right word for these 40 cases.
    naive_result = naive_asof_join_buggy(computed, clean, LEAK_SPEC)
    naive_violations = find_lookahead_violations(naive_result, LEAK_SPEC)
    assert len(naive_violations) == 0
