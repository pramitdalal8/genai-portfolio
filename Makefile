.PHONY: setup lint run-notebooks

# Needs a Python 3.11 interpreter (PyTorch has no Windows wheels for newer interpreters
# yet) and the CPU-only torch build, which isn't a plain PyPI dependency.
setup:
	py -3.11 -m venv .venv311
	.venv311/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	.venv311/Scripts/python -m pip install -e .
	.venv311/Scripts/python -m ipykernel install --user --name py311-genai --display-name "Python 3.11 (GenAI)"

lint:
	.venv311/Scripts/python -m ruff check .
	.venv311/Scripts/python -m mypy .

# Executes and saves outputs in place.
run-notebooks:
	.venv311/Scripts/python scripts/run_notebooks.py
