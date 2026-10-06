import pandas as pd

from ptf.batch_pandas import compute_batch_feature
from ptf.feature_spec import FeatureDefinition

SPEC = FeatureDefinition("trailing_sum_3", "qty", "instrument_id", "t", "sum", 3)


def test_rolling_sum_hand_computed():
    df = pd.DataFrame({
        "instrument_id": [1, 1, 1, 1, 2, 2],
        "t":             [10, 20, 30, 40, 10, 20],
        "qty":           [5, 7, 3, 9, 100, 200],
    })
    out = compute_batch_feature(df, SPEC)
    expected = {
        (1, 10): 5,        # window of 1
        (1, 20): 12,       # 5+7
        (1, 30): 15,       # 5+7+3
        (1, 40): 19,       # 7+3+9 (window caps at 3)
        (2, 10): 100,
        (2, 20): 300,
    }
    for _, row in out.iterrows():
        key = (row["instrument_id"], row["t"])
        assert row[SPEC.name] == expected[key]


def test_shuffled_input_gives_identical_result():
    df = pd.DataFrame({
        "instrument_id": [1, 1, 1, 1, 2, 2],
        "t":             [10, 20, 30, 40, 10, 20],
        "qty":           [5, 7, 3, 9, 100, 200],
    })
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    out_a = compute_batch_feature(df, SPEC).sort_values(["instrument_id", "t"]).reset_index(drop=True)
    out_b = compute_batch_feature(shuffled, SPEC).sort_values(["instrument_id", "t"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_output_is_integer_for_sum_agg():
    df = pd.DataFrame({"instrument_id": [1], "t": [10], "qty": [5]})
    out = compute_batch_feature(df, SPEC)
    assert out[SPEC.name].dtype.kind == "i"
