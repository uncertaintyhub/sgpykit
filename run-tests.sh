xflag=${1:-"-x"}
# PYTHONPATH=$(pwd)/pysparsegrids pytest -v -x test

# requires pytest-benchmark package
PYTHONPATH=$(pwd) pytest --benchmark-skip $xflag -v tests

# benchmarks
# PYTHONPATH=$(pwd) pytest --benchmark-only tests
