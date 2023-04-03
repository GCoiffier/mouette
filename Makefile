install:
	pip3 install -e .
	
wheel:
	python3 setup.py sdist bdist_wheel
	
test:
	pytest --cov=mouette tests/

servedoc:
	mkdocs serve