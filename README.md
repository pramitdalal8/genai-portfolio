# GenAI Portfolio

[![CI](https://github.com/pramitdalal8/genai-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/pramitdalal8/genai-portfolio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Four notebooks on the LLM side of things: prompting, embeddings/RAG, a responsible-AI
audit, and LoRA fine-tuning. Everything runs locally, no API keys needed.

Sibling repos: [data-science-ml-portfolio](https://github.com/pramitdalal8/data-science-ml-portfolio),
[deep-learning-portfolio](https://github.com/pramitdalal8/deep-learning-portfolio).

## Notebooks

1. Prompt engineering — zero-shot, few-shot, structured output
2. Embeddings & RAG
3. Responsible GenAI — safety + fairness auditing
4. Fine-tuning — LoRA / QLoRA

`src/portfolio_common/` is the same small fairness-check package used by
data-science-ml-portfolio's explainability notebook. It's about 15 lines, so it's
duplicated here rather than pulled in as a cross-repo dependency.

## Running it

Needs Python 3.11, same reason as deep-learning-portfolio (PyTorch's Windows wheel
situation).

```bash
make setup
make lint
make run-notebooks
```
