"""Guard the public notebook surface and the explicit release file list."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_check", ROOT / "tools" / "release_check.py")
release_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_check)


def test_release_allowlist_passes():
    assert len(release_check.validate_release(ROOT)) >= 20


@pytest.fixture
def minimal_release(tmp_path):
    """Create text-only inputs for the release validator, not an executable repo."""
    names = sorted(release_check.REQUIRED_FILES)
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Example text\n", encoding="utf-8")
    (tmp_path / "PUBLIC_FILES.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return tmp_path


def test_required_named_text_files_are_accepted(minimal_release):
    paths = release_check.validate_release(minimal_release)
    assert {path.relative_to(minimal_release).as_posix() for path in paths} == release_check.REQUIRED_FILES


@pytest.mark.parametrize("name", ["LICENSE", ".gitignore", ".github/workflows/ci.yml"])
def test_missing_required_file_is_rejected(minimal_release, name):
    (minimal_release / name).unlink()
    with pytest.raises(ValueError, match="Missing, external, or symbolic-link file"):
        release_check.validate_release(minimal_release)


@pytest.mark.parametrize("name", ["LICENSE", ".gitignore", ".github/workflows/ci.yml"])
def test_missing_required_allowlist_entry_is_rejected(minimal_release, name):
    manifest = minimal_release / "PUBLIC_FILES.txt"
    names = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line != name]
    manifest.write_text("\n".join(names) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Required release files missing from allowlist"):
        release_check.validate_release(minimal_release)


@pytest.mark.parametrize("name", ["UNREVIEWED", "nested/LICENSE"])
def test_other_extensionless_paths_are_rejected(minimal_release, name):
    path = minimal_release / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Example text\n", encoding="utf-8")
    manifest = minimal_release / "PUBLIC_FILES.txt"
    manifest.write_text(manifest.read_text(encoding="utf-8") + name + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Non-text release payload"):
        release_check.validate_release(minimal_release)


def test_license_still_rejects_binary_payloads(minimal_release):
    (minimal_release / "LICENSE").write_bytes(b"Example\x00binary")
    with pytest.raises(ValueError, match="Binary payload found"):
        release_check.validate_release(minimal_release)


def test_unlisted_text_file_is_rejected(minimal_release):
    (minimal_release / "unreviewed.txt").write_text("Example text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unreviewed file outside the allowlist"):
        release_check.validate_release(minimal_release)


def test_release_zip_contains_exactly_reviewed_files(tmp_path):
    archive = tmp_path / "public-release.zip"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "release_check.py"), "--zip", str(archive)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    names = (ROOT / "PUBLIC_FILES.txt").read_text(encoding="utf-8").splitlines()
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"quant-research-portfolio/" + name for name in names}
        for name in names:
            assert bundle.read("quant-research-portfolio/" + name) == (ROOT / name).read_bytes()


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
