PROJECT="fddbenchmark"

test:
	pytest

pretty:
	black .
	isort .
