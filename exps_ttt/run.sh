#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=xsum.lora.init.4.comb
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=100g
#SBATCH --cpus-per-task=5
#SBATCH --time=0
##SBATCH --array=0

module load gcc/8.3.0
module load vim tmux cuda/11.1-1
source activate tride

# tir cluster
export TRANSFORMERS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
export HF_DATASETS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
export HF_METRICS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
cache_dir=/home/chuntinz/tir5/pretrain_models/huggingface

# max cluster
export TRANSFORMERS_CACHE=pretrain_models/huggingface
export HF_DATASETS_CACHE=pretrain_models/huggingface
export HF_METRICS_CACHE=pretrain_models/huggingface
cache_dir=pretrain_models/huggingface
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# wandb env variables
export WANDB_PROJECT=gaogao
export WANDB_WATCH="false"

export TOKENIZERS_PARALLELISM="false"
DATE=`date +%Y%m%d`

dataset="super_glue"
subset="rte"
pbsz=2
testset_name="validation"

peft="prompt_tuning"
pL=3
lr=3e-5
lr_scheduler_type="polynomial"
max_steps=50
max_epochs=50
log_steps=10

test_mode="ttt_t0"
model="bigscience/T0pp"

exp_name=${test_mode}.${dataset}.${subset}.${testset_name}
SAVE=checkpoints/${dataset}/${DATE}/${exp_name}
rm -rf ${SAVE}; mkdir -p ${SAVE}
cp ${0} ${SAVE}/run.sh

deepspeed --num_gpus=4 examples/pytorch/t0-zero-shot/run_t0.py \
  --deepspeed deepspeed_configs/ds_config.json \
  --dataset_name ${dataset} --subset_name ${subset} --prompt_set_name ${dataset} --testset_name ${testset_name} \
  --model_name_or_path ${model} --per_device_train_batch_size ${pbsz}  --per_device_eval_batch_size 10 \
  --test_mode ${test_mode} --cache_dir ${cache_dir} \
  --peft_option ${peft} --prompt_tuning_L ${pL} \
  --do_train --logging_steps ${log_steps} --num_train_epochs ${max_epochs} --max_steps ${max_steps} \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-6 \
  --learning_rate ${lr} \
  --lr_scheduler_type ${lr_scheduler_type} \
  --output_dir ${SAVE} --overwrite_output_dir --fp16 --report_to "none" \
  --disable_tqdm "True" 2>&1 | tee ${SAVE}/log.txt