#! /bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"

for host in "$@"; do
    scp $SCRIPT_DIR/eu_tpu_vm_setup_nfs.sh $host:~/
    scp $PROJECT_DIR/tpu_requirements.txt $host:~/
    ssh $host '~/eu_tpu_vm_setup_nfs.sh && pip install -r ~/tpu_requirements.txt' &
done

wait
