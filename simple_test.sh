if [ $# -ge 2 ]; then
    karg="::$2"
else
    karg=""
fi
python -m pytest "$1$karg" -x -v
