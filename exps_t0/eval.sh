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

use_ds="False"

dname="rte" # cb, wsc, copa, wic, anli_r1, anli_r2, anli_r3, winogrande, story_cloze, hellaswag

metric="accuracy"
if [ ${dname} = "rte" ]; then
  dataset="super_glue"
  subset="rte"
  testset_name="validation"
elif [ ${dname} = "cb" ]; then
  dataset="super_glue"
  subset="cb"
  testset_name="validation"
elif [ ${dname} = "anli_r1" ]; then
  dataset="anli"
  subset="none"
  testset_name="dev_r1"
elif [ ${dname} = "anli_r2" ]; then
  dataset="anli"
  subset="none"
  testset_name="dev_r2"
elif [ ${dname} = "anli_r3" ]; then
  dataset="anli"
  subset="none"
  testset_name="dev_r3"
elif [ ${dname} = "wsc" ]; then
  dataset="super_glue"
  subset="wsc.fixed"
  testset_name="validation"
elif [ ${dname} = "winogrande" ]; then
  dataset="winogrande"
  subset="winogrande_xl"
  testset_name="validation"
elif [ ${dname} = "copa" ]; then
  dataset="super_glue"
  subset="copa"
  testset_name="validation"
elif [ ${dname} = "hellaswag" ]; then
  dataset="hellaswag"
  subset="none"
  testset_name="validation"
elif [ ${dname} = "story_cloze" ]; then
  dataset="story_cloze"
  subset="2016"
  testset_name="validation"
elif [ ${dname} = "wic" ]; then
  dataset="super_glue"
  subset="wic"
  testset_name="validation"
else
  echo "wrong dataset name!"
  exit
fi

test_mode="t0"
mname="accuracy"
model="T0_3B"
bsz=100

#model="bigscience/T0pp"
#bsz=1

exp_name=${test_mode}.${dataset}.${subset}.${testset_name}
SAVE=checkpoints/${dataset}/${DATE}/${exp_name}
rm -rf ${SAVE}; mkdir -p ${SAVE}

#CUDA_VISIBLE_DEVICES=0 python -u 
#deepspeed --num_gpus 4 
CUDA_VISIBLE_DEVICES=0 python -u examples/pytorch/t0-zero-shot/run_t0.py \
  --dataset_name ${dataset} --subset_name ${subset} --prompt_set_name ${dataset} --testset_name ${testset_name} \
  --model_name_or_path ${model} --per_device_train_batch_size 1  --per_gpu_eval_batch_size ${bsz} \
  --use_deepspeed ${use_ds} --metric_name ${mname} \
  --test_mode ${test_mode} --cache_dir ${cache_dir} \
  --output_dir ${SAVE} --overwrite_output_dir \
  --disable_tqdm "True" 2>&1 | tee ${SAVE}/log.txt
