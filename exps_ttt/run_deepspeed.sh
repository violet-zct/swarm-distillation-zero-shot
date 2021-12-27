#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=ct.exp
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=50g
#SBATCH -p gpu
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
##SBATCH --array=0

module load gcc/8.3.0
module load vim tmux cuda/11.1-1
source activate ttt

# tir cluster
export TRANSFORMERS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
export HF_DATASETS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
export HF_METRICS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
cache_dir=/home/chuntinz/tir5/pretrain_models/huggingface

# max cluster
root=/home1/xuezhema/projects/ttt-t0-transformers
export TRANSFORMERS_CACHE=${root}/pretrain_models/huggingface
export HF_DATASETS_CACHE=${root}/pretrain_models/huggingface
export HF_METRICS_CACHE=${root}/pretrain_models/huggingface
cache_dir=${root}/pretrain_models/huggingface
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# wandb env variables
export WANDB_PROJECT=gaogao
export WANDB_WATCH="false"

export TOKENIZERS_PARALLELISM="false"
DATE=`date +%Y%m%d`

# dataset=super_glue, subset=rte, cb, wsc.fixed, copa, wic
# dataset=anli, subset=none, testset_name=dev_r1, dev_r2, dev_r3
# dataset=winogrande, subset=winogrande_xl
# dataset=story_cloze, subset=2016, not from huggingface datasets, local download
# dataset=hallaswag

#SLURM_ARRAY_TASK_ID=5
datasets=(wsc winogrande anli_r1 anli_r2 anli_r3 cb rte copa hellaswag story_cloze wic)
dname=${datasets[$SLURM_ARRAY_TASK_ID]} # cb, wsc, copa, wic, anli_r1, anli_r2, anli_r3, winogrande, story_cloze, hellaswag

#dname="rte" # cb, wsc, copa, wic, anli_r1, anli_r2, anli_r3, winogrande, story_cloze, hellaswag

ga=16 # #epochs * #samples/max_steps:control for 50 epochs for low, 16 epochs for medium and 16 for large, upper bound is 24
max_steps=1000
metric="accuracy"
if [ ${dname} = "rte" ]; then
  # 277
  dataset="super_glue"
  subset="rte"
  testset_name="validation"
  ga=14
elif [ ${dname} = "cb" ]; then
  # 57
  dataset="super_glue"
  subset="cb"
  testset_name="validation"
  ga=3
elif [ ${dname} = "anli_r1" ]; then
  # 1000
  dataset="anli"
  subset="none"
  testset_name="dev_r1"
  ga=16
elif [ ${dname} = "anli_r2" ]; then
  # 1000
  dataset="anli"
  subset="none"
  testset_name="dev_r2"
  ga=16
elif [ ${dname} = "anli_r3" ]; then
  # 1200
  dataset="anli"
  subset="none"
  testset_name="dev_r3"
  ga=19
elif [ ${dname} = "wsc" ]; then
  # 104
  dataset="super_glue"
  subset="wsc.fixed"
  testset_name="validation"
  ga=5
elif [ ${dname} = "winogrande" ]; then
  # 1267
  dataset="winogrande"
  subset="winogrande_xl"
  testset_name="validation"
  ga=20
elif [ ${dname} = "copa" ]; then
  # 100
  dataset="super_glue"
  subset="copa"
  testset_name="validation"
  ga=5
elif [ ${dname} = "hellaswag" ]; then
  # 10042
  dataset="hellaswag"
  subset="none"
  testset_name="validation"
  max_steps=2000
  ga=24
elif [ ${dname} = "story_cloze" ]; then
  # 1871
  dataset="story_cloze"
  subset="2016"
  testset_name="validation"
  ga=24
elif [ ${dname} = "wic" ]; then
  # 637
  dataset="super_glue"
  subset="wic"
  testset_name="validation"
  ga=10
else
  echo "wrong dataset name!"
  exit
fi

bsz=1
nprompts=5
eval_bsz=100

peft="lora"
pL=1
lora_pos="encdec"
lora_dropout=0.3
lora_alpha=2

lr=2e-5
lr_scheduler_type="polynomial"
max_epochs=50
eval_steps=50
log_steps=10
debugsize=-1

# used when loss=entropy
temp=1.0
copt="uniform"

test_mode="ttt_t0"
train_data="validation"  # validation, train, stream
model="T0_3B"
# consistency, token_level_entropy, entropy, consistency_pseudo_train, pseudo_train
loss_opt='consistency_pseudo_train'
loss_opt='pseudo_train'
jsd=0
detach_kl_left=1
detach_kl_right=0
ensemble='avg_prob'  # avg_prob, marjority_vote
pseudo_weight=1.0
pseudo_dist="smooth" # smooth (marginalized self-training), argmax
split_answer=0  # 0 for use buggy L1 or only use L2

exp_name=11B_${test_mode}.train.source.${train_data}.${dataset}.${subset}.${testset_name}.${model}.peft.${peft}.lora_alpha${lora_alpha}.lora_drop${lora_dropout}.bn${pL}.sepa${split_answer}.lopt.${loss_opt}.pd.${pseudo_dist}.ens.${ensemble}.sg${sg}.pw${pseudo_weight}.np${nprompts}.bsz${bsz}.ga${ga}.lr${lr}.steps.${max_steps}
SAVE=checkpoints/${dname}/${exp_name}_${DATE}
rm -rf ${SAVE}; mkdir -p ${SAVE}
cp ${0} ${SAVE}/run.sh

#deepspeed --num_gpus=1
#python -u
#python -m torch.distributed.launch --nproc_per_node 4
#CUDA_VISIBLE_DEVICES=0
# python -u examples/pytorch/t0-zero-shot/run_t0.py \
deepspeed --master_addr="192.168.1.1" --master_port=15206 examples/pytorch/t0-zero-shot/run_t0.py \
  --deepspeed deepspeed_configs/ds_config_zero2.json \
  --dataset_name ${dataset} --subset_name ${subset} --prompt_set_name ${dataset} --testset_name ${testset_name} \
  --model_name_or_path ${model} --per_device_train_batch_size ${bsz}  --per_device_eval_batch_size ${eval_bsz} \
  --test_mode ${test_mode} --cache_dir ${cache_dir} --metric_name ${metric} \
  --debug_size ${debugsize} \
  --peft_option ${peft} --bottleneck_dim ${pL} \
  --do_train --logging_steps ${log_steps} --num_train_epochs ${max_epochs} --max_steps ${max_steps} \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-6 \
  --learning_rate ${lr} --evaluation_strategy "steps" --eval_steps ${eval_steps} \
  --loss_option ${loss_opt} --jsd ${jsd} --detach_kl_left ${detach_kl_left} --detach_kl_right ${detach_kl_right} \
  --ensemble_option ${ensemble}  --pseudo_train_loss_weight ${pseudo_weight} --pseudo_dist ${pseudo_dist} \
  --lora_dropout ${lora_dropout} --lora_alpha ${lora_alpha} --lora_pos ${lora_pos} \
  --prob_temperature ${temp} --combine_option ${copt} \
  --train_random_n_prompts ${nprompts} --train_data_source ${train_data} --split_answer_groups ${split_answer} \
  --save_strategy "no" --warmup_steps 100 --gradient_accumulation_steps ${ga} \
  --lr_scheduler_type ${lr_scheduler_type} \
  --output_dir ${SAVE} --overwrite_output_dir --report_to "none" \
  --bf16 \
  --disable_tqdm "True" 2>&1 | tee ${SAVE}/log.txt
