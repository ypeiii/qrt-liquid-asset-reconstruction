"""Guard the public notebook surface and the explicit release file list."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_check", ROOT / "tools" / "release_check.py")
release_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_check)


def test_release_allowlist_passes():
    assert len(release_check.validate_release(ROOT)) >= 20


def test_notebook_output_is_rejected():
    document = json.loads((ROOT / "notebooks" / "01_two_stage_ridge.ipynb").read_text())
    cell = next(cell for cell in document["cells"] if cell["cell_type"] == "code")
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "example"}]
    with pytest.raises(ValueError, match="outputs"):
        release_check.validate_notebook(document)


def test_notebook_attachments_are_rejected():
    document = json.loads((ROOT / "notebooks" / "01_two_stage_ridge.ipynb").read_text())
    document["cells"][0]["attachments"] = {"hidden": {}}
    with pytest.raises(ValueError, match="attachments"):
        release_check.validate_notebook(document)


def test_replacing_reviewed_evidence_is_rejected():
    name = "docs/evidence/public-leaderboard.png"
    payload = (ROOT / name).read_bytes()
    with pytest.raises(ValueError, match="changed"):
        release_check.validate_reviewed_image(name, payload + b"unreviewed")


def test_unreviewed_images_are_rejected():
    with pytest.raises(ValueError, match="not been reviewed"):
        release_check.validate_reviewed_image("docs/evidence/unreviewed.png", b"example")
