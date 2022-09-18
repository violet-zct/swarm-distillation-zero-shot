#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=learnlab
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
#SBATCH --array=0-50
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

datasets=("mrpc" "qqp" "labeled_final" "ARC-Challenge" "ARC-Easy" "hotpotqa" "trivia_qa" "web_questions" "wiki_qa" "dbidaf" "dbert" "droberta" "SelfRC" "ParaphraseRC" "ropes" "squad_v2" "record" "quoref" "tydiqa" "cos_e" "cosmos_qa" "dream" "openbookqa" "qasc" "quail" "quarel" "quartz" "race_high" "race_middle" "sciq" "social_i_qa" "boolq" "multirc" "wiki_hop" "wiqa" "piqa" "amazon_polarity" "app_reviews" "imdb" "rotten_tomatoes" "yelp_review_full" "common_gen" "wiki_bio" "cnn_dailymail" "gigaword" "multi_news" "xsum" "samsum" "ag_news" "dbpedia_14" "trec")
dname=${datasets[$SLURM_ARRAY_TASK_ID]} # cb, wsc, copa, wic, anli_r1, anli_r2, anli_r3, winogrande, story_cloze, hellaswag

metric="accuracy"


if [ ${dname} = "mrpc" ]; then 
  dataset="glue" #0
  subset="mrpc"
  testset_name="validation"
elif [ ${dname} = "qqp" ]; then 
  dataset="glue" #1
  subset="qqp"
  testset_name="validation"
elif [ ${dname} = "labeled_final" ]; then 
  dataset="paws" #2
  subset="labeled_final"
  testset_name="validation"
elif [ ${dname} = "ARC-Challenge" ]; then 
  dataset="ai2_arc" #3
  subset="ARC-Challenge"
  testset_name="validation"
elif [ ${dname} = "ARC-Easy" ]; then 
  dataset="ai2_arc" #4
  subset="ARC-Easy"
  testset_name="validation"
elif [ ${dname} = "hotpotqa" ]; then
  dataset="kilt_tasks" #5
  subset="hotpotqa"
  testset_name="validation"
elif [ ${dname} = "trivia_qa" ]; then
  dataset="trivia_qa" #6
  subset="unfiltered"
  testset_name="validation"
elif [ ${dname} = "web_questions" ]; then
  dataset="web_questions" #7
  subset="none"
  testset_name="test"
elif [ ${dname} = "wiki_qa" ]; then
  dataset="wiki_qa" #8
  subset="none"
  testset_name="validation"
elif [ ${dname} = "dbidaf" ]; then
  dataset="adversarial_qa" #9
  subset="dbidaf"
  testset_name="validation"
elif [ ${dname} = "dbert" ]; then
  dataset="adversarial_qa" #10
  subset="dbert"
  testset_name="validation"
elif [ ${dname} = "droberta" ]; then
  dataset="adversarial_qa" #11
  subset="droberta"
  testset_name="validation"
elif [ ${dname} = "SelfRC" ]; then
  dataset="duorc" #12
  subset="SelfRC"
  testset_name="validation"
elif [ ${dname} = "ParaphraseRC" ]; then
  dataset="duorc"  #13
  subset="ParaphraseRC"
  testset_name="validation"
elif [ ${dname} = "ropes" ]; then
  dataset="ropes" #14
  subset="none"
  testset_name="validation"
elif [ ${dname} = "squad_v2" ]; then
  dataset="squad_v2" #15
  subset="none"
  testset_name="validation"
elif [ ${dname} = "record" ]; then
  dataset="super_glue" #16
  subset="record"
  testset_name="validation"
elif [ ${dname} = "quoref" ]; then
  dataset="quoref" #17
  subset="none"
  testset_name="validation"
elif [ ${dname} = "tydiqa" ]; then
  dataset="tydiqa" #18
  subset="primary_task"
  testset_name="validation"
elif [ ${dname} = "cos_e" ]; then
  dataset="cos_e" #19
  subset="v1.11"
  testset_name="validation"
elif [ ${dname} = "cosmos_qa" ]; then
  dataset="cosmos_qa" #20
  subset="none"
  testset_name="validation"
elif [ ${dname} = "dream" ]; then
  dataset="dream" #21
  subset="none"
  testset_name="validation"
elif [ ${dname} = "openbookqa" ]; then
  dataset="openbookqa" #22
  subset="main"
  testset_name="validation"
elif [ ${dname} = "qasc" ]; then
  dataset="qasc" #23
  subset="none"
  testset_name="validation"
elif [ ${dname} = "quail" ]; then
  dataset="quail" #24
  subset="none"
  testset_name="validation"
elif [ ${dname} = "quarel" ]; then
  dataset="quarel" #25
  subset="none"
  testset_name="validation"
elif [ ${dname} = "quartz" ]; then
  dataset="quartz" #26
  subset="none"
  testset_name="validation"
elif [ ${dname} = "race_high" ]; then
  dataset="race" #27
  subset="high"
  testset_name="validation"
elif [ ${dname} = "race_middle" ]; then
  dataset="race" #28
  subset="middle"
  testset_name="validation"
elif [ ${dname} = "sciq" ]; then
  dataset="sciq" #29
  subset="none"
  testset_name="validation"
elif [ ${dname} = "social_i_qa" ]; then
  dataset="social_i_qa" #30
  subset="none"
  testset_name="validation"
elif [ ${dname} = "boolq" ]; then
  dataset="super_glue" #31
  subset="boolq"
  testset_name="validation"
elif [ ${dname} = "multirc" ]; then
  dataset="super_glue" #32
  subset="multirc"
  testset_name="validation"
elif [ ${dname} = "wiki_hop" ]; then
  dataset="wiki_hop" #33
  subset="original"
  testset_name="validation"
elif [ ${dname} = "wiqa" ]; then
  dataset="wiqa" #34
  subset="none"
  testset_name="validation"
elif [ ${dname} = "piqa" ]; then
  dataset="piqa" #35
  subset="none"
  testset_name="validation"
elif [ ${dname} = "amazon_polarity" ]; then
  dataset="amazon_polarity" #36
  subset="none"
  testset_name="test"
elif [ ${dname} = "app_reviews" ]; then
  dataset="app_reviews" #37
  subset="none"
  testset_name="train"
elif [ ${dname} = "imdb" ]; then
  dataset="imdb" #38
  subset="none"
  testset_name="test"
elif [ ${dname} = "rotten_tomatoes" ]; then
  dataset="rotten_tomatoes" #39
  subset="none"
  testset_name="validation"
elif [ ${dname} = "yelp_review_full" ]; then
  dataset="yelp_review_full" #40
  subset="none"
  testset_name="test"
elif [ ${dname} = "common_gen" ]; then
  dataset="common_gen" #41
  subset="none"
  testset_name="validation"
elif [ ${dname} = "wiki_bio" ]; then
  dataset="wiki_bio" #42
  subset="none"
  testset_name="val"
elif [ ${dname} = "cnn_dailymail" ]; then
  dataset="cnn_dailymail" #43
  subset="3.0.0"
  testset_name="validation"
elif [ ${dname} = "gigaword" ]; then
  dataset="gigaword" #44
  subset="none"
  testset_name="validation"
elif [ ${dname} = "multi_news" ]; then
  dataset="multi_news" #45
  subset="none"
  testset_name="validation"
elif [ ${dname} = "xsum" ]; then
  dataset="xsum" #46
  subset="none"
  testset_name="validation"
elif [ ${dname} = "samsum" ]; then
  dataset="samsum" #47
  subset="none"
  testset_name="validation"
elif [ ${dname} = "ag_news" ]; then
  dataset="ag_news" #48
  subset="none"
  testset_name="test"
elif [ ${dname} = "dbpedia_14" ]; then
  dataset="dbpedia_14" #49
  subset="none"
  testset_name="test"
elif [ ${dname} = "trec" ]; then
  dataset="trec" #50
  subset="none"
  testset_name="test"
else
  echo "wrong dataset name!"
  exit
fi


if [ ${testset_name} = "train" ]; then
  exit
fi

test_mode="t0"
mname="accuracy"
model="T0"
bsz=50

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
