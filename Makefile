install:
	pip3 install -e .
	
wheel:
	python3 setup.py sdist bdist_wheel
	
test:
	python -m pytest --cov=mouette tests/

servedoc:
	python -m mkdocs serve
