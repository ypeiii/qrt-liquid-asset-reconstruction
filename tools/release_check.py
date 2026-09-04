"""Validate a text allowlist plus reviewed evidence and build a clean public ZIP.

This is a guardrail, not a proof that a strategy or secret has not been exposed.
Manually review every allowlisted file and any additions before publication.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


TEXT_SUFFIXES = {".md", ".py", ".toml", ".yml", ".ipynb", ".txt"}
REVIEWED_IMAGES = {
    "docs/evidence/public-leaderboard.png": "57678e19c522fcbbfed92348c5278c8c6ddd4646426e898fe83f3de1d3495743",
}
IGNORED_PARTS = {".git", ".venv", "artifacts", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXPOSURE_PATTERNS = {
    "absolute Windows path": re.compile(r"\b[A-Za-z]:[\\/]"),
    "absolute personal POSIX path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "email address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "common access-token prefix": re.compile(r"\b(?:ghp|github_pat|sk-proj)_[A-Za-z0-9_-]{16,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "non-English CJK text": re.compile(r"[\u3400-\u9fff]"),
}


def validate_reviewed_image(name: str, payload: bytes) -> None:
    """Permit only the exact unmodified screenshot that was manually reviewed.

    Adding or replacing an image requires a new visual and metadata review,
    followed by a deliberate digest update. Arbitrary image payloads are banned.
    """
    if name not in REVIEWED_IMAGES:
        raise ValueError("Image has not been reviewed for publication.")
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Reviewed evidence must retain its original PNG format.")
    if hashlib.sha256(payload).hexdigest() != REVIEWED_IMAGES[name]:
        raise ValueError("Reviewed evidence image changed; review it before publishing.")


def validate_notebook(document: dict) -> None:
    if set(document) != {"cells", "metadata", "nbformat", "nbformat_minor"}:
        raise ValueError("Unexpected notebook top-level payload.")
    metadata = document.get("metadata", {})
    if set(metadata).difference({"kernelspec", "language_info"}):
        raise ValueError("Unexpected notebook metadata.")
    if metadata.get("kernelspec") != {"display_name": "Python 3", "language": "python", "name": "python3"}:
        raise ValueError("Unexpected kernel metadata.")
    if metadata.get("language_info") != {"name": "python"}:
        raise ValueError("Unexpected language metadata.")
    for cell in document.get("cells", []):
        allowed = {"cell_type", "id", "metadata", "source"}
        if cell.get("cell_type") == "code":
            allowed |= {"outputs", "execution_count"}
            if cell.get("outputs") != [] or cell.get("execution_count") is not None:
                raise ValueError("Notebook outputs and execution counts must be cleared.")
        elif cell.get("cell_type") != "markdown":
            raise ValueError("Unexpected notebook cell type.")
        if set(cell) != allowed or cell.get("metadata") != {}:
            raise ValueError("Cell metadata, attachments, or extra payloads are not allowed.")


def validate_release(root: Path) -> list[Path]:
    root = root.resolve()
    names = [line.strip() for line in (root / "PUBLIC_FILES.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)) or "PUBLIC_FILES.txt" not in names:
        raise ValueError("Release allowlist must be unique and include itself.")
    paths = []
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Allowlist entries must be safe relative POSIX paths.")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing, external, or symbolic-link file: {name}")
        if name in REVIEWED_IMAGES:
            validate_reviewed_image(name, path.read_bytes())
            paths.append(path)
            continue
        if path.name != ".gitignore" and path.suffix not in TEXT_SUFFIXES:
            raise ValueError(f"Non-text release payload: {name}")
        contents = path.read_text(encoding="utf-8")
        if "\x00" in contents:
            raise ValueError(f"Binary payload found in {name}")
        for label, pattern in EXPOSURE_PATTERNS.items():
            if pattern.search(contents):
                raise ValueError(f"Review required: {label} found in {name}")
        if path.suffix == ".ipynb":
            validate_notebook(json.loads(contents))
        paths.append(path)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.as_posix() not in names:
            raise ValueError(f"Unreviewed file outside the allowlist: {relative.as_posix()}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="Optional new allowlist-only archive, outside the repository.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = validate_release(root)
    print(f"PASS: {len(paths)} allowlisted files; reviewed image hashes and notebook outputs/metadata checked.")
    if args.zip:
        destination = args.zip.resolve()
        if destination.is_relative_to(root):
            raise ValueError("Write the public archive outside the source repository.")
        if destination.exists():
            raise FileExistsError("Choose a new archive path; an existing archive is never overwritten.")
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(paths):
                info = zipfile.ZipInfo("quant-research-portfolio/" + path.relative_to(root).as_posix())
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
        print("Archive SHA-256:", hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
