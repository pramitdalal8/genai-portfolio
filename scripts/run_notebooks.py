"""Execute every notebook in this repo and save real outputs back into the .ipynb files.

Requires the project kernel to be registered first (see the `setup` Makefile target).

Usage (run from the repo root):
    python scripts/run_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 900
KERNEL = "py311-genai"

NOTEBOOKS = [
    "genai/notebooks/01_prompt_engineering.ipynb",
    "genai/notebooks/02_embeddings_and_rag.ipynb",
    "genai/notebooks/03_responsible_genai.ipynb",
    "genai/notebooks/04_finetuning_and_qlora.ipynb",
]


def run_notebook(relative_path: str) -> None:
    path = REPO_ROOT / relative_path
    print(f"[run_notebooks] executing {relative_path}")
    nb = nbformat.read(path, as_version=4)
    executor = ExecutePreprocessor(timeout=TIMEOUT_SECONDS, kernel_name=KERNEL)
    executor.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)
    print(f"[run_notebooks] done {relative_path}")


def main() -> int:
    failures: list[tuple[str, Exception]] = []
    for relative_path in NOTEBOOKS:
        try:
            run_notebook(relative_path)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"[run_notebooks] FAILED {relative_path}: {exc}", file=sys.stderr)
            failures.append((relative_path, exc))

    if failures:
        print(f"\n[run_notebooks] {len(failures)} notebook(s) failed:", file=sys.stderr)
        for relative_path, error in failures:
            print(f"  - {relative_path}: {error}", file=sys.stderr)
        return 1

    print("\n[run_notebooks] all notebooks executed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
