"""Evidence path resolution must reject missing and unsafe paths."""
from src.dashboard.evidence import resolve_evidence_path


def test_valid_evidence_file(tmp_path):
    db_path = tmp_path / "surveillance.db"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    photo = evidence / "intrusion_sample.jpg"
    photo.write_bytes(b"jpeg")
    assert resolve_evidence_path(str(db_path), photo.name) == photo.resolve()


def test_reject_path_traversal(tmp_path):
    db_path = tmp_path / "surveillance.db"
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"data")
    assert resolve_evidence_path(str(db_path), "../secret.jpg") is None
    assert resolve_evidence_path(str(db_path), str(outside)) is None


def test_missing_evidence_does_not_break_dashboard(tmp_path):
    assert resolve_evidence_path(str(tmp_path / "surveillance.db"), "missing.jpg") is None
    assert resolve_evidence_path(str(tmp_path / "surveillance.db"), None) is None
