#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
echo $SCRIPT_DIR
echo $PROJECT_DIR

export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
echo $PYTHONPATH
export WANDB_API_KEY=''

export experiment_name=''
export experiment_project=''
export experiment_note="$experiment_name"

python3 -m apt.pretrain_main \
    --logging.online=False \
    --logging.prefix="$experiment_name" \
    --logging.project="$experiment_project" \
    --logging.output_dir="$HOME/experiment_output/$experiment_name" \
    --logging.random_delay=0.0 \
    --logging.notes="$experiment_note"
