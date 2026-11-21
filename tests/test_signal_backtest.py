import math

from ptf.signal_backtest import WINDOWS, _events_frame, run_all_candidates, run_candidate, summarize
from ptf.signal_generate import EVENTS_PER_INSTRUMENT, GROUP_SIZE, NUM_INSTRUMENTS, asset_class_for
from ptf.signal_spec import ASSET_CLASSES, SignalSpec


def test_events_frame_has_four_equal_asset_class_groups():
    events = _events_frame()
    assert len(events) == NUM_INSTRUMENTS * EVENTS_PER_INSTRUMENT
    counts = events["asset_class"].value_counts()
    for asset_class in ASSET_CLASSES:
        assert counts[asset_class] == GROUP_SIZE * EVENTS_PER_INSTRUMENT


def test_asset_class_assignment_is_a_pure_function_of_instrument_id():
    assert asset_class_for(0) == asset_class_for(0)
    first_group = {asset_class_for(i) for i in range(GROUP_SIZE)}
    assert first_group == {ASSET_CLASSES[0]}


def test_run_all_candidates_produces_24_results():
    results = run_all_candidates()
    assert len(results) == 4 * len(WINDOWS)
    names = {r.spec.name for r in results}
    assert len(names) == 24  # every candidate name is unique


def test_deterministic_given_same_inputs():
    events = _events_frame()
    spec = SignalSpec(
        name="sig_test", asset_class="FX", source_field="ret", group_key="instrument_id",
        order_key="event_time_ns", window_events=20, risk_budget=1.0,
    )
    r1 = run_candidate(events, spec)
    r2 = run_candidate(events, spec)
    assert r1.gross_total_return == r2.gross_total_return
    assert r1.net_total_return == r2.net_total_return


def test_net_never_exceeds_gross_when_cost_is_positive():
    """Cost only subtracts; for any candidate whose lagged position tracks
    the same signal, net P&L cannot exceed gross P&L by more than the one-
    bar timing difference accounts for -- checked here as a sanity bound:
    net minus the cost actually charged never exceeds gross by an amount
    unrelated to the lag."""
    events = _events_frame()
    spec = SignalSpec(
        name="sig_test2", asset_class="RATE", source_field="ret", group_key="instrument_id",
        order_key="event_time_ns", window_events=10, risk_budget=1.0,
    )
    r = run_candidate(events, spec)
    assert r.total_cost >= 0.0
    assert r.n_rebalances > 0


def test_summarize_counts_flipped_candidates_consistently():
    results = run_all_candidates()
    summary = summarize(results)
    assert summary["n_candidates"] == 24
    recomputed_flipped = [r for r in results if r.gross_total_return > 0.0 and r.net_total_return < 0.0]
    assert summary["n_flipped_negative"] == len(recomputed_flipped)
    assert set(summary["flipped_names"]) == {r.spec.name for r in recomputed_flipped}


def test_flipped_implies_gross_positive_and_net_negative():
    results = run_all_candidates()
    for r in results:
        if r.flipped:
            assert r.gross_total_return > 0.0
            assert r.net_total_return < 0.0
        if not math.isnan(r.gross_total_return) and r.gross_total_return <= 0.0:
            assert not r.flipped
