#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=w3
#SBATCH --job-name=eval.11b
##SBATCH --comment="NeurIPS 2022 rebuttal deadline"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --exclude 'a100-st-p4d24xlarge-459,a100-st-p4d24xlarge-465'
#SBATCH --mem=0GB
#SBATCH --signal=USR1@90
#SBATCH --open-mode=append
#SBATCH --time=2:00:00
##SBATCH --time=0
#SBATCH --array=0-10
#SBATCH --wckey=submitit


# command
export MODULEPATH=/data/home/vkhalidov/modulefiles:$MODULEPATH
module load cuda/11.3
module load nccl/2.12.7-cuda.11.3
module load nccl_efa/1.2.0-nccl.2.12.7-cuda.11.3
export SUBMITIT_EXECUTOR=slurm

source activate t0

export TRANSFORMERS_CACHE=pretrain_models/huggingface
export HF_DATASETS_CACHE=pretrain_models/huggingface
export HF_METRICS_CACHE=pretrain_models/huggingface
cache_dir=pretrain_models/huggingface
#export TRANSFORMERS_OFFLINE=0
export WANDB_MODE=offline

# wandb env variables
#export WANDB_PROJECT=gaogao
#export WANDB_WATCH="false"

DATE=`date +%Y%m%d`

#SLURM_ARRAY_TASK_ID=0
use_ds="False"
datasets=(wsc winogrande anli_r1 anli_r2 anli_r3 cb rte copa hellaswag story_cloze wic)
dname=${datasets[$SLURM_ARRAY_TASK_ID]} # cb, wsc, copa, wic, anli_r1, anli_r2, anli_r3, winogrande, story_cloze, hellaswag

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
  testset_name="train"
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
model="T0"
bsz=100
if [ ${dname} = "hellaswag" ]; then
	bsz=50
fi

#model="bigscience/T0pp"
#bsz=1

exp_name=${test_mode}.${dataset}.${subset}.${testset_name}
SAVE=results

#CUDA_VISIBLE_DEVICES=0 python -u 
#deepspeed --num_gpus 4 
python -u examples/pytorch/t0-zero-shot/run_t0.py \
  --dataset_name ${dataset} --subset_name ${subset} --prompt_set_name ${dataset} --testset_name ${testset_name} \
  --model_name_or_path ${model} --per_device_train_batch_size 1  --per_device_eval_batch_size ${bsz} \
  --use_deepspeed ${use_ds} --metric_name ${mname} \
  --test_mode ${test_mode} --cache_dir ${cache_dir} \
  --output_dir ${SAVE} --overwrite_output_dir --cb_surgery 0 \
  --disable_tqdm "True"
