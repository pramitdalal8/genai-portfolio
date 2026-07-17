# GenAI Portfolio

Generative AI and LLMs: 4 fully-executed notebooks — prompt engineering, embeddings &
RAG, responsible GenAI auditing, and LoRA fine-tuning — all fully local (no API keys,
no paid calls).

Sibling repos: [data-science-ml-portfolio](https://github.com/pramitdalal8/data-science-ml-portfolio),
[deep-learning-portfolio](https://github.com/pramitdalal8/deep-learning-portfolio).

## Notebooks

- **[genai/](genai)** — prompt engineering, embeddings & RAG, responsible GenAI (safety
  + fairness auditing), LoRA fine-tuning.

## Shared library

- **[src/portfolio_common/](src/portfolio_common)** — small, real, tested utilities
  (currently a fairness/bias auditing check) used by the responsible-GenAI notebook,
  installed as an editable package rather than loaded via file-path hacks. (Also used
  by [data-science-ml-portfolio](https://github.com/pramitdalal8/data-science-ml-portfolio) — kept
  as a small self-contained copy here rather than a cross-repo dependency.)

## Quick start

```bash
make setup            # needs Python 3.11 (PyTorch has no Windows wheels for newer interpreters yet)
make lint               # ruff + mypy
make run-notebooks        # executes the notebooks in place, saves real outputs
```

## Repository layout

- [genai/](genai) — notebooks
- [src/portfolio_common/](src/portfolio_common) — shared, tested utility package
- [docs/](docs) — a concept-to-dataset reference spanning DS/ML, deep learning, and GenAI

## Design decisions

Depth over breadth: 4 fully-executed notebooks with real local models beat a larger
set of templates that were never run. Every claim (real embeddings & RAG, real
fairness auditing, real LoRA fine-tuning) is backed by code that runs, executed in CI
(`.github/workflows/notebook-check.yml` validates structure on every PR;
`notebooks-full-run.yml` executes every notebook, triggered manually or weekly).
