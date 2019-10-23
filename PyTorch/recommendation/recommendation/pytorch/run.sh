#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

THRESHOLD=1.0
BASEDIR='../'
DATASET=${DATASET:-ml-20m}

# Get command line seed
seed=${1:-1}

# Get the multipliers for expanding the dataset
USER_MUL=${USER_MUL:-1}
ITEM_MUL=${ITEM_MUL:-1}

DATASET_DIR=${BASEDIR}/${DATASET}x${USER_MUL}x${ITEM_MUL}

if [ -d ${DATASET_DIR} ]
then

	python ncf.py ${DATASET_DIR} \
        -l 0.0002 \
        -b 65536 \
        --layers 256 256 128 64 \
        -f 64 \
		--seed $seed \
        --threshold $THRESHOLD \
        --user_scaling ${USER_MUL} \
        --item_scaling ${ITEM_MUL} \
        --cpu_dataloader \
        --epochs 1\
        --random_negatives

	echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi





