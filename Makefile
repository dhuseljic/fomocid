.PHONY: docs docs-strict

docs:
	python -m sphinx -b html docs docs/_build/html

docs-strict:
	python -m sphinx -W --keep-going -b html docs docs/_build/html
