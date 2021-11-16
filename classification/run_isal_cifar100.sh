#!/bin/bash

"""use the standard influence function(use validation dataset to calculate left grad) to mining
"""

if [ $# -lt 3 ]; then
    echo "USAGE: $0 WORK_DIR DATA_DIR CONFIG..."
    exit
fi


WORK_DIR=$1 #/workspace/
DATA_DIR=$2 #/data/cifar-10-batches-py
CONFIG_PATH=$3 #pycls/configs/archive/cifar/resnet/R-110_nds_1gpu.yaml

STEP=${STEP:-50}
TRAIN_STEP=${TRAIN_STEP:-10} # data ratio from 1/50 to 10/50

testset_file=${DATA_DIR}/test

RATIOS=()
for ((i=0;i<=${STEP}-1;i++))
do
    sum=`expr $STEP - $i`
    RATIOS[i]=1/${sum}
done

TIMESTAMP=$(date "+%Y%m%d%H%M%S")
mkdir -p ${WORK_DIR}/${TIMESTAMP}

export PYTHONPATH="$(dirname $0)":$PYTHONPATH
for ((i=0;i<${TRAIN_STEP};i++))
do
    budget_ratio=${RATIOS[i]}
    workdir_step=${WORK_DIR}/${TIMESTAMP}/step_${i}
    mkdir -p ${workdir_step}
    unlabeled_file=${workdir_step}/unlabeled_data.pickle
    labeled_file=${workdir_step}/labeled_data.pickle

    # training on labeled data and mining
    # first step without mining
    if [ ${i} == 0 ]; then
        python $(dirname "$0")/src/train.py --config-path=${CONFIG_PATH} \
            --data-dir=${DATA_DIR} --work-dir=${workdir_step} \
            --budget-ratio=${budget_ratio} ${@:4}
    else 
        j=`expr ${i} - 1`
        prev_workdir_step=${WORK_DIR}/${TIMESTAMP}/step_${j}
        prev_labeled_file=${prev_workdir_step}/labeled_data.pickle
        prev_unlabeled_file=${prev_workdir_step}/unlabeled_data.pickle

        python $(dirname "$0")/src/train.py --config-path=${CONFIG_PATH} \
            --data-dir=${DATA_DIR} --work-dir=${workdir_step} \
            --budget-ratio=${budget_ratio} \
            --prev-model-result=${prev_workdir_step}/'result_on_unlabeled.pickle' \
            --prev-labeled-file=${prev_labeled_file} --prev-unlabeled-file=${prev_unlabeled_file} ${@:4}
    fi

    # testing on unlabeled data with trained model in current step and save predicitions
    # skip the last ratio(no data in unlabled annotation file)
    if [ ${i} != `expr ${#RATIOS[@]} - 1` ]; then
        python $(dirname "$0")/src/test_calc_val_grad.py --config-path=${CONFIG_PATH} \
            --data-dir=${DATA_DIR} --work-dir=${workdir_step} \
            --test-data-file=${testset_file} --model-path=${workdir_step}/'model.pth' \
            --out-path=${workdir_step}

        python $(dirname "$0")/src/test_calc_s_test.py --config-path=${CONFIG_PATH} \
            --data-dir=${DATA_DIR} --work-dir=${workdir_step} \
            --test-data-file=${labeled_file} --model-path=${workdir_step}/'model.pth' \
            --out-path=${workdir_step} --in-path=${workdir_step}

        python $(dirname "$0")/src/test_calc_influ.py --config-path=${CONFIG_PATH} \
            --data-dir=${DATA_DIR} --work-dir=${workdir_step} \
            --test-data-file=${unlabeled_file} --model-path=${workdir_step}/'model.pth' \
            --in-path=${workdir_step}  
    fi
done

#plot all step results
python $(dirname "$0")/src/tools/plot_results.py \
    --work-dir=${WORK_DIR}/${TIMESTAMP} --train-step=${TRAIN_STEP} 
