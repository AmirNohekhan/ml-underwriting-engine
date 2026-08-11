.PHONY: install generate train score all test api

install:
	python -m pip install -e .[dev]

generate:
	underwriting-engine generate

train:
	underwriting-engine train

score:
	underwriting-engine score

all:
	underwriting-engine all

test:
	pytest

api:
	uvicorn underwriting_engine.api:app --reload
