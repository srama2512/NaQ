#!/bin/bash

source ~/enable_naq.sh

cd $NAQ_ROOT

DEVICE=$1
TASK_TYPE=$2
SPLIT=$3
FEAT_TYPE=$4
PRETRAINED_PATH=$5
PRETRAINED_DIR="$(dirname "${PRETRAINED_PATH}")"

if [[ $SPLIT == "train" ]]; then
    EVAL_GT_JSON="data/nlq_train.json"
elif [[ $SPLIT == "val" ]]; then
    EVAL_GT_JSON="data/nlq_val.json"
else
    EVAL_GT_JSON=''
fi

if [[ $TASK_TYPE == "nlq" ]]; then
    TASK="nlq_official_v1"
elif [[ $TASK_TYPE == "nlq_v2" ]]; then
    TASK="nlq_official_v2"
elif [[ $TASK_TYPE == "tacos" ]]; then
    TASK="tacos_official_v1"
fi

v_feat_types=${FEAT_TYPE}_clip
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(./data/features/slowfast)
    (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
  vsldataset_fv="slowfast"
fi
if [[ ${v_feat_types} == *"egovlp"*  ]]; then
  v_feat_dirs+=(./data/features/egovlp)
    (( v_feat_dim += 256  ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
  vsldataset_fv="egovlp"
fi
if [[ ${v_feat_types} == *"internvideo"*  ]]; then
  v_feat_dirs+=(./data/features/internvideo)
    (( v_feat_dim += 2304  ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
  vsldataset_fv="internvideo"
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(./data/features/clip)
  (( v_feat_dim += 512 ))
fi


CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONPATH=$PYTHONPATH:./ReLER/ python ./ReLER/ms_cm/inference_ego4d_slowfast.py \
    --resume $PRETRAINED_PATH \
    --v_feat_dirs ${v_feat_dirs[@]} \
    --vsldataset_task ${TASK} \
    --vsldataset_fv ${vsldataset_fv} \
    --vslnet_datapath ./data/ \
    --split $SPLIT \
    --v_feat_dim $v_feat_dim \
    --testing_gt_json "$EVAL_GT_JSON" \
    --eval_results_dir $PRETRAINED_DIR
