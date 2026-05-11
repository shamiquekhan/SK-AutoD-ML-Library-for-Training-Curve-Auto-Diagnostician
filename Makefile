PYTHON ?= python

.PHONY: test lint format build clean

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check sk_autod tests

format:
	$(PYTHON) -m ruff format sk_autod tests

build:
	$(PYTHON) -m build

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(path, ignore_errors=True) for path in [pathlib.Path('build'), pathlib.Path('dist')] if path.exists()]; [shutil.rmtree(path, ignore_errors=True) for path in pathlib.Path('.').glob('*.egg-info')]"
