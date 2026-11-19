PY ?= .venv/bin/python

.PHONY: check lint test hooks site

check: lint test

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

test:
	$(PY) -m pytest -q

hooks:
	git config core.hooksPath .githooks

site:
	$(PY) -m gaitpinn.site --runs ../runs --out ../site
