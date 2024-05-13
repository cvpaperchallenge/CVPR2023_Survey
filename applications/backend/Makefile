.PHONY: ruff-format
ruff-format:
	poetry run ruff format src tests

.PHONY: ruff-format-check
ruff-format-check:
	poetry run ruff format --check src tests

.PHONY: ruff-lint
ruff-lint:
	poetry run ruff check src tests --fix

.PHONY: ruff-lint-check
ruff-lint-check:
	poetry run ruff check src tests

.PHONY: mypy
mypy:
	poetry run mypy src

.PHONY: test
test:
	poetry run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: format
format:
	$(MAKE) ruff-lint
	$(MAKE) ruff-format

.PHONY: lint
lint:
	$(MAKE) ruff-lint-check
	$(MAKE) ruff-format-check
	$(MAKE) mypy

.PHONY: test-all
test-all:
	$(MAKE) lint
	$(MAKE) test