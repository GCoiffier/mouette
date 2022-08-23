language: python
python:
  - "3.10"
before_install:
  - pip install pipenv
  - pip install pytest pytest-cov
  - pip install coveralls
install:
  - pip install aenum numpy scipy osqp tqdm
  - pip install -e .
script:
  - pytest --cov=mouette tests/
after_success:
  - coveralls

