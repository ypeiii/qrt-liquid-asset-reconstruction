# Publication checklist

The public repository is [ypeiii/qrt-liquid-asset-reconstruction](https://github.com/ypeiii/qrt-liquid-asset-reconstruction). Use this checklist for updates and redistributions. Publication does not independently verify a leaderboard claim.

1. Review the included original screenshot and `RESULTS.md`: rank 5, public score 0.7511, and display name are visible; 484 and the screenshot capture time are not. Do not relabel it as a final ranking.
2. Check the challenge-specific sharing conditions. [Platform terms](https://challengedata.ens.fr/terms_of_use) distinguish provider data licenses and participant reports/code; individual challenge terms can matter. This repository does not include competition data. No legal clearance or permission to release the private strategy is implied.
3. Review every file in `PUBLIC_FILES.txt`. Never copy the private notebook directory, its old Git history, anonymous factor matrices, original outputs, caches, model files, predictions, Optuna databases, backups, or credentials into this directory.
4. Run `python -m pytest -q`, `python tools/run_notebooks.py`, and `python tools/release_check.py`. The latter checks the explicit file list, notebook output/metadata policy, text payloads, the reviewed screenshot digest, and common exposure patterns. Automated checks do not replace a manual strategy-disclosure review.
5. Preserve the [MIT License](../LICENSE) for the author's original code and documentation and the [third-party notices](../THIRD_PARTY_NOTICES.md). The screenshot is excluded from that MIT grant; attribution does not establish permission to reuse third-party content. No private competition assets are distributed or licensed by this release.
6. Publish updates using **only** the allowlisted files. Do not reuse a private `.git` directory. Review the staged diff before each commit. A `.gitignore` does not remove previously committed content.
7. After publishing, check GitHub's rendered notebooks, run the configured CI, and add the actual repository URL to the CV. Do not invent a URL or a passing badge before it exists.

`python tools/release_check.py --zip ../quant-research-portfolio-public.zip` packages only the approved file list. The original leaderboard screenshot is the sole approved binary. Any new or replaced image requires visual and metadata review, an allowlist entry, and a deliberate digest update. Original notebook attachments remain prohibited.

## Upload using GitHub's website

For this project, upload updates to the [existing repository](https://github.com/ypeiii/qrt-liquid-asset-reconstruction); a new repository is not needed. A public repository is accessible to reviewers without an invitation. If creating a separate copy, avoid generating duplicate README, ignore, or license files: this release already supplies them. [GitHub repository creation guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository).

Extract the release ZIP into a new folder. Upload the contents of its `quant-research-portfolio` directory, not the ZIP or its outer wrapper. `README.md`, `pyproject.toml`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `PUBLIC_FILES.txt` should sit at the repository root. Retain all subdirectories, including `.github`, and the `.gitignore` file. Check these dot-prefixed paths explicitly: a manual selection can omit them. Use the repository's upload option, review the proposed paths, and commit the files. [GitHub file-upload guide](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

Check the rendered README, four notebooks, evidence image, and Actions result before linking the project on a CV. Upload only this extracted release; do not drag in the private working directory. The release contains fewer than GitHub's 100-file browser upload limit, with each file well below its 25 MiB limit.

## What remains private

The full feature construction process and its outputs remain outside this directory. There is no encrypted, compiled, serialized, or hidden copy of that process in the public package. The unimplemented `PrivateFeatureProvider` is an explicit boundary, not a secretly callable implementation.

Disclosing model families, grouped validation, and ensemble selection still shares high-level research ideas. If even those ideas are confidential, do not publish this repository. Once published, copies and forks may persist even if the original is deleted.
