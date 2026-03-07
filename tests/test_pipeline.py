from types import SimpleNamespace

import pandas as pd

from src import cli
from src.JudeaPearl.DataAnalyst import DataAnalyst


def test_checkpoint_requires_matching_scm_signature():
    signature = '{"scenario":"fresh"}'
    checkpoint = {"scm_signature": signature}

    assert cli._checkpoint_matches_scm(checkpoint, signature) is True
    assert cli._checkpoint_matches_scm({}, signature) is False
    assert cli._checkpoint_matches_scm({"scm_signature": '{"scenario":"stale"}'}, signature) is False


def test_history_checkpoint_requires_matching_metadata():
    signature = '{"scenario":"fresh"}'
    history_data = {
        "histories": [1, 2],
        "scm_signature": signature,
        "sample_count": 2,
        "max_interactions": 3,
        "temp_subject": 0.0,
    }

    assert cli._history_checkpoint_matches(history_data, signature, 2, 3, 0.0) is True
    assert cli._history_checkpoint_matches(history_data, signature, 3, 3, 0.0) is False
    assert cli._history_checkpoint_matches(history_data, signature, 2, 4, 0.0) is False


def test_data_analyst_returns_false_when_sem_estimation_fails(tmp_path, monkeypatch):
    analyst = DataAnalyst(
        final_df=pd.DataFrame({"outcome": [1.0, 2.0]}),
        meta_data={},
        variable_mapping={"outcome": "outcome"},
        edge_dict={},
        interaction_data={},
        scm_simple={
            "Variable1": {
                "__class__": "EndogenousVariable",
                "name": "outcome",
                "scenario_description": "test scenario",
                "agents_in_scenario": ["agent"],
                "variable_type": "continuous",
                "units": "units",
                "levels": [],
            }
        },
    )

    monkeypatch.setattr(
        analyst,
        "estimate_sem",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="lavaan failed", stdout=""),
    )

    assert analyst.analyze_data(str(tmp_path), str(tmp_path)) is False
