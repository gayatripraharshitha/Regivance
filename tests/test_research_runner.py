from scripts import run_research


def test_controlled_outputs_are_archived_not_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(run_research, "ROOT", tmp_path)
    tables = tmp_path / "reports/tables"
    tables.mkdir(parents=True)
    old = tables / "lstm_qlike_objective_predictions.csv"
    old.write_text("previous controlled experiment", encoding="utf-8")
    unrelated = tables / "user_notes.csv"
    unrelated.write_text("keep me", encoding="utf-8")
    archive = tmp_path / "archive"
    run_research.archive_qlike_outputs(archive)
    assert (archive / old.name).read_text() == "previous controlled experiment"
    assert not old.exists()
    assert unrelated.read_text() == "keep me"
