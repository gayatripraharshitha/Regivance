from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from src.research_config import CONFIG, load_research_config
from src.evaluation.final_protocol import final_evaluation_plan, history_digest


def small_protocol():
    cfg = deepcopy(CONFIG)
    cfg["evaluation"].update(train_size=150, validation_size=42)
    cfg["final_evaluation"].update(freeze_date="2021-01-01", exposed_data_through="2020-12-31", observations=21)
    cfg["models"].update(epochs=1, hidden_size=2)
    return cfg


def sample_dates():
    return pd.bdate_range(end="2021-01-01", periods=300).append(pd.bdate_range("2021-01-04", periods=50))


def test_final_waits_for_all_targets():
    cfg = small_protocol()
    plan = final_evaluation_plan(sample_dates()[:325], cfg)
    assert plan["status"] == "awaiting_future_data"
    assert plan["missing_observations"] == 1
    assert final_evaluation_plan(sample_dates()[:326], cfg)["status"] == "ready"


def test_final_first_dates_fixed_after_new_rows_arrive():
    cfg = small_protocol()
    a = final_evaluation_plan(sample_dates()[:326], cfg)
    b = final_evaluation_plan(sample_dates(), cfg)
    assert a == b
    assert a["first_position"] == 300
    assert a["raw_stop_position"] == 326
    assert a["start_date"] == "2021-01-04"


def test_final_embargo_prevents_label_overlap():
    cfg = small_protocol()
    for split in final_evaluation_plan(sample_dates(), cfg)["splits"]:
        assert split.train.stop - 1 + 5 < split.validation.start
        assert split.validation.stop - 1 + 5 < split.test.start


def test_final_excludes_exposed_dates_even_if_freeze_earlier():
    cfg = small_protocol()
    cfg["final_evaluation"]["exposed_data_through"] = "2021-01-08"
    plan = final_evaluation_plan(sample_dates(), cfg)
    assert plan["start_date"] == "2021-01-11"


@pytest.mark.parametrize("index", [pd.Index([1, 2]), pd.DatetimeIndex(["2020-01-01", None]),
    pd.DatetimeIndex(["2020-01-01", "2020-01-01"]), pd.DatetimeIndex(["2020-01-02", "2020-01-01"])])
def test_final_rejects_invalid_dates(index):
    with pytest.raises(ValueError):
        final_evaluation_plan(index, small_protocol())


def test_history_digest_is_append_stable_but_revision_sensitive():
    data = pd.DataFrame({"a": [0.01, 0.02, 0.03]}, index=pd.date_range("2020-01-01", periods=3))
    original = history_digest(data.iloc[:2], "2020-01-02")
    assert original == history_digest(data, "2020-01-02")
    data.iloc[0, 0] = 0.04
    assert original != history_digest(data, "2020-01-02")


@pytest.mark.parametrize("section,key,value", [("evaluation", "embargo", 4),
    ("evaluation", "step_size", 1), ("evaluation", "horizon", True),
    ("features", "n_regimes", 4), ("models", "epochs", 0),
    ("final_evaluation", "observations", 22), ("final_evaluation", "freeze_date", "1920-01-01")])
def test_config_rejects_invalid_protocol(tmp_path, section, key, value):
    cfg = deepcopy(CONFIG)
    cfg[section][key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_research_config(path)


def test_prospective_all_models_real_synthetic_smoke():
    # Synthetic integration test, never reported as a real-market final result.
    from scripts.final_evaluation import evaluate
    cfg = small_protocol()
    dates = sample_dates()
    rng = np.random.default_rng(42)
    common = rng.normal(0, 0.01, (len(dates), 1))
    returns = pd.DataFrame(common + rng.normal(0, 0.005, (len(dates), 4)), index=dates, columns=list("ABCD"))
    predictions, choices, training = evaluate(returns, cfg, final_evaluation_plan(dates, cfg))
    assert predictions.index.equals(dates[300:321])
    models = predictions.drop(columns=["actual", "fold", "regime"])
    assert models.shape == (21, 11)
    assert np.isfinite(models.to_numpy()).all()
    assert (models > 0).all().all()
    assert len(choices) == 3 and len(training) == 1


def test_registration_and_pending_gate_do_not_fit_models(tmp_path, monkeypatch, capsys):
    import scripts.final_evaluation as command
    cfg = small_protocol()
    raw = tmp_path / "source.csv"
    raw.write_text("synthetic fixture")
    returns = pd.DataFrame({"A": np.full(299, 0.01)}, index=sample_dates()[:299])
    monkeypatch.setattr(command, "FREEZE", tmp_path / "frozen.json")
    monkeypatch.setattr(command, "load_research_config", lambda: cfg)
    monkeypatch.setattr(command, "load_industry_returns", lambda path: returns)
    monkeypatch.setattr(command, "code_hashes", lambda: {"model.py": "original"})
    monkeypatch.setattr(command, "evaluate", lambda *args: pytest.fail("Cannot fit without future data"))
    monkeypatch.setattr(command.sys, "argv", ["final_evaluation.py", "--freeze", "--protocol", str(command.FREEZE), "--raw", str(raw)])
    command.main()
    assert command.FREEZE.is_file()
    with pytest.raises(ValueError, match="immutable"):
        command.main()
    monkeypatch.setattr(command.sys, "argv", ["final_evaluation.py", "--run", "--protocol", str(command.FREEZE), "--raw", str(raw)])
    command.main()
    assert "awaiting_future_data" in capsys.readouterr().out
    monkeypatch.setattr(command, "code_hashes", lambda: {"model.py": "edited"})
    with pytest.raises(ValueError, match="Code/config differs"):
        command.main()
    monkeypatch.setattr(command, "code_hashes", lambda: {"model.py": "original"})
    returns.iloc[0, 0] = 0.02
    with pytest.raises(ValueError, match="Historical source returns changed"):
        command.main()
