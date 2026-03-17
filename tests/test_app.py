import ast
from pathlib import Path

import numpy as np
import pandas as pd


APP_PATH = Path(__file__).resolve().parents[1] / "stuffplus-app" / "app.py"


def _load_functions(*names):
    """Load selected function definitions from app.py without executing the Streamlit app body."""
    source = APP_PATH.read_text()
    tree = ast.parse(source)

    selected_nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]

    module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace = {"pd": pd, "np": np}
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return [namespace[name] for name in names]


build_usage_splits, build_pitch_finder_table, get_numeric_slider_bounds = _load_functions(
    "build_usage_splits", "build_pitch_finder_table", "get_numeric_slider_bounds"
)


# Checks that usage splits only include pitches meeting the threshold and reports formatted percentages.
def test_build_usage_splits_filters_and_formats_percentages():
    player_df = pd.DataFrame(
        {
            "pitch_type": ["FF", "FF", "FF", "SL", "SL", "CH"],
            "batter_handedness": ["L", "R", "L", "L", "R", "R"],
        }
    )

    result = build_usage_splits(
        player_df,
        pitch_order=["SL", "FF", "CH"],
        min_pitches=2,
    )

    expected = pd.DataFrame(
        [
            {"Pitch": "SL", "Overall": "33.3%", "vs LHH": "33.3%", "vs RHH": "33.3%"},
            {"Pitch": "FF", "Overall": "50.0%", "vs LHH": "66.7%", "vs RHH": "33.3%"},
        ]
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


# Checks that pitch finder aggregation computes rounded metrics, usage %, and handedness labels.
def test_build_pitch_finder_table_aggregates_and_rounds_output():
    df = pd.DataFrame(
        {
            "game_year": [2025, 2025, 2025, 2025],
            "PlayerName": ["Pitcher A", "Pitcher A", "Pitcher A", "Pitcher A"],
            "pitch_type": ["FF", "FF", "FF", "SL"],
            "HB_obs": [10, 12, 11, 5],
            "iVB_obs": [15.2, 15.4, 15.5, 8.1],
            "arm_angle": [45.2, 45.4, 45.3, 44.0],
            "release_speed": [95.1, 95.3, 95.2, 84.0],
            "release_spin_rate": [2400, 2450, 2500, 2100],
            "release_extension": [6.1, 6.3, 6.2, 5.8],
            "Stuff+_pt": [110.0, 112.0, 111.0, 95.0],
            "spin_axis": [180, 182, 181, 200],
            "ssw_in": [1.2, 1.4, 1.3, 0.8],
            "ssw_x": [0.1, 0.2, 0.1, 0.0],
            "ssw_z": [0.3, 0.4, 0.3, 0.2],
            "spin_efficiency": [0.91, 0.93, 0.92, 0.88],
            "is_lefty": [1, 1, 1, 1],
        }
    )

    result = build_pitch_finder_table(df, min_pitches=2)

    assert list(result["Pitch"]) == ["FF"]
    row = result.iloc[0]
    assert row["Pitcher"] == "Pitcher A"
    assert row["Handedness"] == "LHP"
    assert row["Pitches"] == 3
    assert row["Usage %"] == 75.0
    assert row["Stuff+"] == 111.0
    assert row["Velo"] == 95.2
    assert row["iVB"] == 15.4
    assert row["HB"] == 11.0
    assert row["Arm Angle"] == 45.3
    assert row["Extension"] == 6.2
    assert row["Spin"] == 2450.0
    assert row["Spin Axis"] == 181.0
    assert row["Spin Eff."] == 0.92
    assert row["SSW"] == 1.3
    assert row["SSW_X"] == 0.1
    assert row["SSW_Z"] == 0.3


# Checks integer-like numeric values produce integer slider bounds with unit steps.
def test_get_numeric_slider_bounds_for_integer_like_series():
    result = get_numeric_slider_bounds(pd.Series([10, 12, "11", np.nan]))

    assert result == {
        "min": 10,
        "max": 12,
        "default": (10, 12),
        "step": 1,
        "format": None,
        "kind": "int",
    }


# Checks float ranges produce decimal slider bounds and fine-grained steps for small spans.
def test_get_numeric_slider_bounds_for_float_series_small_span():
    result = get_numeric_slider_bounds(pd.Series([1.04, 1.18, "1.26"]))

    assert result == {
        "min": 1.0,
        "max": 1.3,
        "default": (1.0, 1.3),
        "step": 0.01,
        "format": "%.2f",
        "kind": "float",
    }


# Checks empty or non-varying data does not produce slider bounds.
def test_get_numeric_slider_bounds_returns_none_for_empty_or_constant_values():
    assert get_numeric_slider_bounds(pd.Series([np.nan, None, "x"])) is None
    assert get_numeric_slider_bounds(pd.Series([5, 5, "5"])) is None
