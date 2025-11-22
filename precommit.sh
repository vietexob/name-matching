echo "[0 / 5] setting vars"
export PYPROJECT_MODULE="name_matching"

echo "[1 / 5] removing unused imports and variables"
autoflake -r --remove-all-unused-imports --remove-unused-variables --quiet --in-place $PYPROJECT_MODULE
autoflake -r --remove-all-unused-imports --remove-unused-variables --quiet --in-place tests

echo "[2 / 5] ordering imports"
isort $PYPROJECT_MODULE
isort tests

echo "[3 / 5] formatting code"
black .

echo "[4 / 5] linting code"
pylint $PYPROJECT_MODULE
pylint tests

echo "[5 / 5] test coverage"
coverage run -m pytest tests
coverage report
