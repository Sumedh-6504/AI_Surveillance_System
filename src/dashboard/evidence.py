"""Resolve incident screenshots only from the local evidence directory."""
from pathlib import Path


def resolve_evidence_path(db_path: str, evidence_file):
    """Return an existing JPEG inside the database's evidence folder, otherwise None."""
    if not isinstance(evidence_file, str) or not evidence_file:
        return None
    # Only file basenames written by the local intrusion recorder are allowed.
    if Path(evidence_file).name != evidence_file:
        return None
    if not evidence_file.endswith(".jpg"):
        return None
    root = (Path(db_path).resolve().parent / "evidence").resolve()
    candidate = (root / evidence_file).resolve()
    if candidate.parent != root or not candidate.is_file():
        return None
    return candidate
