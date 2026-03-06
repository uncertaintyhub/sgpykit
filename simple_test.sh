SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ $# -ge 2 ]; then
    karg="::$2"
else
    karg=""
fi
PYTHONPATH=${SCRIPT_DIR} pytest -v -x "${SCRIPT_DIR}/$1$karg"
