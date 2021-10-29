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


source activate tride
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

DATE=`date +%Y%m%d`

dataset="anli"
test_mode="t0"
testset_name="test_r1"
model="bigscience/T0_3B"

exp_name={test_mode}.${dataset}.${testset_name}
SAVE=checkpoints/${dataset}/${DATE}/${exp_name}
rm -rf ${SAVE}; mkdir -p ${SAVE}

deepspeed examples/pytorch/t0-zero-shot/run_t0.py \
  --deepspeed deepspeed_configs/ds_config.json --dataset_name ${dataset}\
  --model_name_or_path ${model} --per_device_train_batch_size 1   \
  --test_mode ${test_mode} --cache_dir ${cache_dir} \
  --output_dir ${SAVE} --overwrite_output_dir --fp16 \
  --disable_tqdm "True" 2>&1 | tee ${SAVE}/log.txt