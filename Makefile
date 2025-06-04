SHELL := /bin/sh

install:
	poetry install

test:
	poetry run pytest src/test

lint:
	poetry run pre-commit run --show-diff-on-failure --color=always --all-files

hooks:
	poetry run pre-commit install --install-hooks

install dataset:
    python src/data/prepare_data.py --data_dir ./CIFAR10
