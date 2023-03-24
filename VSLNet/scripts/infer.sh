#!/bin/bash

source ~/enable_naq.sh

cd $NAQ_ROOT

DEVICES=$1
TASK_TYPE=$2
SPLIT=$3
FEAT_TYPE=$4
EXPT_ROOT=$5

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
    MAX_POS_LEN=128
    if [[ $SPLIT == "train" ]]; then
        EVAL_GT_JSON="data/nlq_train.json"
    elif [[ $SPLIT == "val" ]]; then
        EVAL_GT_JSON="data/nlq_val.json"
    else
        EVAL_GT_JSON=''
    fi
elif [[ $TASK_TYPE == "nlq_v2" ]]; then
    TASK="nlq_official_v2"
    MAX_POS_LEN=128
    if [[ $SPLIT == "train" ]]; then
        EVAL_GT_JSON="data/nlq_train_v2.json"
    elif [[ $SPLIT == "val" ]]; then
        EVAL_GT_JSON="data/nlq_val_v2.json"
    else
        EVAL_GT_JSON=''
    fi
elif [[ $TASK_TYPE == "tacos" ]]; then
    TASK="tacos_official_v1"
    MAX_POS_LEN=512
    if [[ $SPLIT == "train" ]]; then
        EVAL_GT_JSON="data/tacos_train.json"
    elif [[ $SPLIT == "val" ]]; then
        EVAL_GT_JSON="data/tacos_val.json"
    else
        EVAL_GT_JSON=''
    fi
fi

CUDA_VISIBLE_DEVICES="$DEVICES" python VSLNet/main.py \
 --task $TASK \
 --predictor bert \
 --mode $SPLIT \
 --video_feature_dim $FEAT_DIM \
 --max_pos_len $MAX_POS_LEN \
 --fv $FV \
 --model_dir $EXPT_ROOT/checkpoints \
 --eval_gt_json "$EVAL_GT_JSON"
