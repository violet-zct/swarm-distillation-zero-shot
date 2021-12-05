#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=rte.val.consistency.3e-5
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=60g
#SBATCH -p isi
#SBATCH --cpus-per-task=6
#SBATCH --time=96:00:00
##SBATCH --array=0

# module load gcc/8.3.0
# module load vim tmux cuda/11.1-1
# source activate tride

# tir cluster
# export TRANSFORMERS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
# export HF_DATASETS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
# export HF_METRICS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
# cache_dir=/home/chuntinz/tir5/pretrain_models/huggingface

# max cluster
export TRANSFORMERS_CACHE=/project/jonmay_231/max/ttt-t0-transformers/pretrain_models/huggingface
export HF_DATASETS_CACHE=/project/jonmay_231/max/ttt-t0-transformers/pretrain_models/huggingface
export HF_METRICS_CACHE=/project/jonmay_231/max/ttt-t0-transformers/pretrain_models/huggingface
cache_dir=/project/jonmay_231/max/ttt-t0-transformers/pretrain_models/huggingface
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# wandb env variables
export WANDB_PROJECT=gaogao
export WANDB_WATCH="false"

export TOKENIZERS_PARALLELISM="false"
DATE=`date +%Y%m%d`

dataset="super_glue"
subset="rte"
bsz=1
nprompts=5
testset_name="validation"

peft="prompt_tuning"
peft="lora"
pL=1
lora_pos="encdec"

lr=3e-5
lr_scheduler_type="polynomial"
max_steps=1000
max_epochs=50
log_steps=10
debugsize=-1

#loss_opt="token_level_entropy"  # consistency, token_level_entropy, entropy
# loss_opt="entropy"
loss_opt=consistency
temp=1.0
copt="uniform"
eval_steps=10

test_mode="ttt_t0"
# train_data="validation"  # test, stream
#train_data="train"  # test, stream
train_data="stream"
model="T0_3B"

exp_name=${test_mode}.train.source.${train_data}.${dataset}.${subset}.${testset_name}.${model}.np${nprompts}.peft.${peft}.bn${pL}.lora_pos.${lora_pos}.lopt.${loss_opt}.combine.${copt}.temp.${temp}.lr.${lr}
SAVE=checkpoints/jh/${dataset}/${DATE}/${exp_name}
rm -rf ${SAVE}; mkdir -p ${SAVE}
cp ${0} ${SAVE}/run.sh

#deepspeed --num_gpus=1
#python -u
#python -m torch.distributed.launch --nproc_per_node 4
#CUDA_VISIBLE_DEVICES=0
python -u examples/pytorch/t0-zero-shot/run_t0.py \
  --dataset_name ${dataset} --subset_name ${subset} --prompt_set_name ${dataset} --testset_name ${testset_name} \
  --model_name_or_path ${model} --per_device_train_batch_size ${bsz}  --per_device_eval_batch_size 10 \
  --test_mode ${test_mode} --cache_dir ${cache_dir} \
  --debug_size ${debugsize} \
  --peft_option ${peft} --bottleneck_dim ${pL} \
  --lora_pos 'dec' --lora_alpha 4 \
  --do_train --logging_steps ${log_steps} --num_train_epochs ${max_epochs} --max_steps ${max_steps} \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-6 \
  --learning_rate ${lr} \
  --loss_option ${loss_opt} \
  --lora_dropout 0.1 --lora_alpha 4 --lora_pos ${lora_pos} \
  --prob_temperature ${temp} --combine_option ${copt} --detach_one_side 1 \
  --train_random_n_prompts ${nprompts} --train_data_source ${train_data} \
  --save_strategy "no" --warmup_steps 100 --gradient_accumulation_steps 16 \
  --evaluation_strategy "steps" --eval_steps ${eval_steps} \
  --lr_scheduler_type ${lr_scheduler_type} \
  --output_dir ${SAVE} --overwrite_output_dir --report_to "none" \
  --disable_tqdm "True" 2>&1 | tee ${SAVE}/log.txt

