#! /bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"

for host in "$@"; do
    echo "Cleanning on host: $host"
    ssh $host 'rm -rf ~/m3ae' &
done

wait
