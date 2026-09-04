"""Execute all four public notebooks without storing outputs in source files."""

import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if len(notebooks) != 4:
        raise RuntimeError("Expected exactly four public notebooks.")
    with tempfile.TemporaryDirectory(prefix="quant-portfolio-smoke-") as scratch:
        temporary = Path(scratch)
        spec_directory = temporary / "kernels" / "portfolio-demo"
        spec_directory.mkdir(parents=True)
        (spec_directory / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Portfolio demo smoke check",
            "language": "python",
            "env": {
                "PORTFOLIO_ARTIFACT_DIR": str(temporary / "artifacts"),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        }), encoding="utf-8")
        manager = KernelSpecManager(kernel_dirs=[str(temporary / "kernels")])
        for path in notebooks:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            kernel = KernelManager(kernel_name="portfolio-demo", kernel_spec_manager=manager)
            client = NotebookClient(
                notebook, timeout=180, km=kernel,
                resources={"metadata": {"path": str(root)}},
                allow_errors=False,
            )
            # Explicit cleanup is needed because the manager is supplied here.
            try:
                client.execute()
            finally:
                if kernel.has_kernel:
                    kernel.shutdown_kernel(now=True)
                kernel.cleanup_resources()
            counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == "code"]
            if not counts or any(count is None for count in counts):
                raise RuntimeError(f"Incomplete notebook execution: {path.name}")
            print(f"PASS {path.name}: {len(counts)} code cells executed", flush=True)
    print("All four notebooks passed; source outputs remain empty.", flush=True)


if __name__ == "__main__":
    main()

