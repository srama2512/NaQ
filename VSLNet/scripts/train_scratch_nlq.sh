#!/bin/bash

source ~/enable_naq.sh

cd $NAQ_ROOT

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))
}
export MAIN_PORT=$(rand 1024 2048)
export MAIN_PORT_TCL=$(rand 1024 2048)

DEVICES=$1
TASK_TYPE=$2
FEAT_TYPE=$3
EXPT_ROOT=$4
LR=$5

mkdir -p $EXPT_ROOT

if [[ $FEAT_TYPE == "internvideo" ]]; then
    FEAT_DIM=2304
    FV="internvideo"
elif [[ $FEAT_TYPE == "egovlp" ]]; then
    FEAT_DIM=256
    FV="egovlp"
elif [[ $FEAT_TYPE == "slowfast" ]]; then
    FEAT_DIM=2304
    FV="slowfast"
fi

if [[ $TASK_TYPE == "nlq" ]]; then
    TASK="nlq_official_v1"
elif [[ $TASK_TYPE == "tacos" ]]; then
    TASK="tacos_official_v1"
fi

CUDA_VISIBLE_DEVICES="$DEVICES" python -W ignore -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port $MAIN_PORT_TCL \
    VSLNet/main.py \
        --task $TASK \
        --predictor bert \
        --dim 128 \
        --mode train \
        --video_feature_dim $FEAT_DIM \
        --max_pos_len 128 \
        --epochs 200 \
        --fv $FV \
        --num_workers 4 \
        --model_dir $EXPT_ROOT/checkpoints \
        --eval_gt_json "data/nlq_val.json" \
        --log_to_tensorboard "nlq" \
        --tb_log_dir $EXPT_ROOT/runs \
        --remove_empty_queries_from train val \
        --batch_size 32 \
        --init_lr $LR
