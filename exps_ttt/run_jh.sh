#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=ct.exp
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=50g
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

bsz=1
ga=16
nprompts=5
eval_bsz=10

peft="lora"
pL=1
lora_pos="encdec"

lr=2e-5
lr_scheduler_type="polynomial"
max_steps=1000
max_epochs=50
eval_steps=50
log_steps=1
debugsize=-1

# used when loss=entropy
temp=1.0
copt="uniform"

test_mode="ttt_t0"
train_data="validation"  # validation, train, stream
model="T0_3B"
# consistency, token_level_entropy, entropy, consistency_pseudo_train, pseudo_train
# loss_opt='consistency_pseudo_train'
loss_opt='pseudo_train'
jsd=0
detach_kl_left=1
detach_kl_right=0
ensemble='avg_prob'  # avg_prob, marjority_vote
pseudo_weight=1.0
pseudo_dist="smooth" # smooth (marginalized self-training), argmax

exp_name=${test_mode}.train.source.${train_data}.${dataset}.${subset}.${testset_name}.${model}.peft.${peft}.bn${pL}.lopt.${loss_opt}.pd.${pseudo_dist}.ens.${ensemble}.sg${sg}.pw${pseudo_weight}.np${nprompts}.bsz${bsz}.ga${ga}.lr${lr}.steps.${max_steps}
SAVE=checkpoints/jh/${dname}/${DATE}/${exp_name}
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
  --lora_dropout 0.1 --lora_alpha 4 --lora_pos ${lora_pos} \
  --prob_temperature ${temp} --combine_option ${copt} \
  --train_random_n_prompts ${nprompts} --train_data_source ${train_data} \
  --save_strategy "no" --warmup_steps 100 --gradient_accumulation_steps ${ga} \
  --lr_scheduler_type ${lr_scheduler_type} \
  --output_dir ${SAVE} --overwrite_output_dir --report_to "none" \
  --disable_tqdm "True" 2>&1 | tee ${SAVE}/log.txt

