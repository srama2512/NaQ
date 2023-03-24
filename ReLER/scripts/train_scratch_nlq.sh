#!/bin/bash

source ~/enable_naq.sh

cd $NAQ_ROOT

DEVICES=$1
TASK_TYPE=$2
FEAT_TYPE=$3
EXPT_ROOT=$4
LR=$5

if [[ $TASK_TYPE == "nlq" ]]; then
    TASK="nlq_official_v1"
elif [[ $TASK_TYPE == "tacos" ]]; then
    TASK="tacos_official_v1"
fi

ctx_mode=video_tef
v_feat_types=${FEAT_TYPE}_clip
results_root=$EXPT_ROOT
task_type=$TASK
EXP_NAME=vlen600_$FEAT_TYPE
bsz=32

######## setup video+text features
feat_root=features

# video features
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

# multi_scale param
scale_list=()
scale_list+=(2)
scale_list+=(3)
scale_list+=(4)
scale_list+=(5)
scale_list+=(6)

# hyperparameters
MAX_V_LEN=600
HIDDEN_SIZE=256
N_CROSS_ENCODER_LAYERS=3
DROPOUT=0.1
USE_SW=1
USE_VS=1
VS_PROB=0.5
CONTRASTIVE_HIDDEN_SIZE=64

#### training
sw_len_ratio=()
sw_len_ratio+=(0.4)
sw_len_ratio+=(0.8)

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))

}
export MAIN_PORT=$(rand 1024 2048)
export MAIN_PORT_TCL=$(rand 1024 2048)

CUDA_VISIBLE_DEVICES=$DEVICES PYTHONPATH=$PYTHONPATH:./ReLER/ python -W ignore -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port $MAIN_PORT_TCL \
     ReLER/ms_cm/train_ego4d_slowfast.py \
        --ctx_mode ${ctx_mode} \
        --v_feat_dirs ${v_feat_dirs[@]} \
        --numscale_list ${scale_list[@]} \
        --v_feat_dim ${v_feat_dim} \
        --t_feat_dim 512 \
        --bsz ${bsz} \
        --eval_bsz 256 \
        --results_root ${results_root} \
        --vsldataset_task ${task_type} \
        --vsldataset_fv ${vsldataset_fv} \
        --no_pin_memory \
        --num_workers 4 \
        --vsldataset_num_workers 16 \
        --vslnet_datapath ./data/ \
        --vslnet_dataset_save_dir ./ReLER/vslnet_dataset_savepath \
        --eval_gt_json ./data/nlq_val.json \
        --no_aux_loss \
        --cross_first \
        --max_v_l $MAX_V_LEN \
        --lw_saliency 1 \
        --lw_highlight 20 \
        --enc_layers 0 \
        --hidden_dim $HIDDEN_SIZE \
        --v_hidden_size $HIDDEN_SIZE \
        --bi_hidden_size $HIDDEN_SIZE \
        --hidden_size $HIDDEN_SIZE \
        --num_cross_encoder_layers $N_CROSS_ENCODER_LAYERS \
        --dropout $DROPOUT \
        --v_hidden_dropout_prob $DROPOUT \
        --hidden_dropout_prob $DROPOUT \
        --v_attention_probs_dropout_prob $DROPOUT \
        --attention_probs_dropout_prob $DROPOUT \
        --vslnet_thres_in_train 0 \
        --use_sw $USE_SW \
        --sw_len_ratio ${sw_len_ratio[@]} \
        --use_vs $USE_VS \
        --vs_prob $VS_PROB \
        --lr $LR \
        --video_frame_contrastive_loss \
        --video_frame_contrastive_loss_coef 1 \
        --contrastive_hdim $CONTRASTIVE_HIDDEN_SIZE \
        --exp_id $EXP_NAME
