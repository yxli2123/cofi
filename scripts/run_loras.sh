#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -A pnlp
#SBATCH -t 11:00:00
TASK=MNLI
SUFFIX=sparsity0.95
EX_CATE=CoFi
PRUNING_TYPE=structured_heads+structured_mlp+hidden+layer
SPARSITY=0.95
DISTILL_LAYER_LOSS_ALPHA=0.9
DISTILL_CE_LOSS_ALPHA=0.1
LAYER_DISTILL_VERSION=4
SPARSITY_EPSILON=0.01
DISTILLATION_PATH=/mnt/lustre/sjtu/home/xc915/superb/CoFiPruning/teacher-model
FINAL_THRESHOLD=0.2
LR_RATIO=0.05


glue_low=(MRPC RTE STSB CoLA)
glue_high=(MNLI QQP QNLI SST2)

proj_dir=.

code_dir=${proj_dir}

# task and data
task_name=$TASK
data_dir=$proj_dir/data/glue_data/${task_name}

# pretrain model
model_name_or_path=bert-base-uncased

# logging & saving
logging_steps=100
save_steps=0


# train parameters
max_seq_length=128
batch_size=32 
learning_rate=2e-5
reg_learning_rate=0.01
epochs=20 

# seed
seed=57

# output dir
ex_name_suffix=$SUFFIX
ex_name=${task_name}_${ex_name_suffix}
ex_cate=$EX_CATE
output_dir=${proj_dir}/out/${task_name}/${ex_cate}/${ex_name}

# pruning and distillation
pruning_type=$PRUNING_TYPE
target_sparsity=$SPARSITY
distillation_path=$DISTILLATION_PATH
distill_layer_loss_alpha=$DISTILL_LAYER_LOSS_ALPHA
distill_ce_loss_alpha=$DISTILL_CE_LOSS_ALPHA
distill_temp=2
# 2: fix hidden layers, 3: min distance matching without restriction, 4: min distance matching with restriction
layer_distill_version=$LAYER_DISTILL_VERSION

scheduler_type=linear


if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=50
    epochs=100
    start_saving_best_epochs=50
    prepruning_finetune_epochs=4
    lagrangian_warmup_epochs=20
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
    prepruning_finetune_epochs=1
    lagrangian_warmup_epochs=2
fi

pretrained_pruned_model=None

# FT after pruning
if [[ $pruning_type == None ]]; then
  pretrained_pruned_model=$SPARSITY_EPSILON
  learning_rate=${11}
  scheduler_type=none
  output_dir=$pretrained_pruned_model/FT-lr${learning_rate}
  epochs=20
  batch_size=64
fi

mkdir -p $output_dir

python3 $code_dir/run_glue_loras.py \
	   --output_dir ${output_dir} \
	   --logging_steps ${logging_steps} \
	   --task_name ${task_name} \
	   --model_name_or_path ${model_name_or_path} \
	   --ex_name ${ex_name} \
	   --do_train \
	   --do_eval \
	   --max_seq_length ${max_seq_length} \
	   --per_device_train_batch_size ${batch_size} \
	   --per_device_eval_batch_size 32 \
	   --learning_rate ${learning_rate} \
	   --reg_learning_rate ${reg_learning_rate} \
	   --num_train_epochs ${epochs} \
	   --overwrite_output_dir \
	   --save_steps ${save_steps} \
	   --eval_steps ${eval_steps} \
	   --evaluation_strategy steps \
	   --seed ${seed} \
	   --pruning_type ${pruning_type} \
     --pretrained_pruned_model ${pretrained_pruned_model} \
     --target_sparsity $target_sparsity \
     --freeze_embeddings \
     --do_distill \
     --do_layer_distill \
     --final_threshold ${FINAL_THRESHOLD} \
     --low_rank_parameter_ratio ${LR_RATIO} \
     --distillation_path $distillation_path \
     --distill_ce_loss_alpha $distill_ce_loss_alpha \
     --distill_loss_alpha $distill_layer_loss_alpha \
     --distill_temp $distill_temp \
     --scheduler_type $scheduler_type \
     --layer_distill_version $layer_distill_version \
     --prepruning_finetune_epochs $prepruning_finetune_epochs \
     --lagrangian_warmup_epochs $lagrangian_warmup_epochs 2>&1 | tee ${output_dir}/log.txt