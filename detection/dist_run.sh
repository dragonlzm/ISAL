#!/bin/bash
if [ $# -lt 3 ]; then
    echo "USAGE: $0 WORK_DIR DATA_DIR CONFIG ..."
    exit
fi


WORK_DIR=$1
DATA_DIR=$2
CONFIG=$3

GPUS=${GPUS:-4}
PORT=${PORT:-29500}
TRAIN_STEP=${TRAIN_STEP:-10}
IMG_NUM=${IMG_NUM:-1000}
INIT_IMG_NUM=${INIT_IMG_NUM:-5000}
DELETE_MODEL=${DELETE_MODEL:-1}

# for pretrained model
rm -rf ${HOME}/.cache
mkdir -p ${DATA_DIR}/.cache
ln -sf ${DATA_DIR}/.cache ${HOME}/.cache

rm -rf /mmdetection/data
ln -sf ${DATA_DIR} /mmdetection/data
TIMESTAMP=$(date "+%Y%m%d%H%M%S")
mkdir -p ${WORK_DIR}/${TIMESTAMP}

export PYTHONPATH="$(dirname $0)":$PYTHONPATH
for ((i=0;i<${TRAIN_STEP};i++))
do
    ratio=${RATIOS[i]}
    workdir=${WORK_DIR}/${TIMESTAMP}/step_${i}
    mkdir -p ${workdir}
    unlabeled_file=${workdir}/unlabeled_datasets.json

    if [ ${i} == 0 ]; then
        python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
            $(dirname "$0")/src/train.py $CONFIG --initial-ratio=${INIT_IMG_NUM} \
            --work-dir=${workdir} --ratio=0 --launcher pytorch ${@:4}
    else
        j=`expr ${i} - 1`
        prev_workdir=${WORK_DIR}/${TIMESTAMP}/step_${j}
        prev_labeled_file=${prev_workdir}/labeled_datasets.json
        prev_unlabeled_file=${prev_workdir}/unlabeled_datasets.json

        python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
            $(dirname "$0")/src/train.py $CONFIG \
            --model-result=${prev_workdir}/results.bbox.json \
            --options data.train.initial.ann_file=${prev_labeled_file} data.train.unlabeled.ann_file=${prev_unlabeled_file} \
            --work-dir=${workdir} --ratio=${IMG_NUM} --launcher pytorch ${@:4}
    fi

    # skip the last ratio(no data in unlabled annotation file)
    if [ ${i} != `expr ${TRAIN_STEP} - 1` ]; then
       python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
           $(dirname "$0")/src/test.py $CONFIG ${workdir}/latest.pth --gpu-collect \
           --options data.test.ann_file=${unlabeled_file} jsonfile_prefix=${workdir}/results \
           --out ${workdir}/result.pkl --format-only --launcher pytorch
    fi
done

if [ ${DELETE_MODEL} == 1 ]; then
    find ${WORK_DIR}/${TIMESTAMP} -name '*.pth' -type f -print -exec rm -rf {} \;
fi