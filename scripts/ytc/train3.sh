export CUDA_VISIBLE_DEVICES=7
export CUDA_LAUNCH_BLOCKING=1

dset_name=ytc
ctx_mode=video_tef
v_feat_types=slowfast
t_feat_type=gpt2
exp_id=exp
results_root=results
device=1
enc_layers=3
dec_layers=3
query_num=30
n_txt_mu=5
n_visual_mu=30

span_loss_type=l1
sim_loss_coef=1
neg_loss_coef=0.5
exp_id=test
seed=2023
lr=1e-4
lr_gamma=0.1
neg_choose_epoch=80
lr_drop=400

######## data paths
train_path=data/ytc/train.jsonl
eval_path=data/ytc/val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../ytc_features


# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/ytc_slowfast_features/)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi

# text features
t_feat_dir=${feat_root}/ytc_gpt2_feats/
t_feat_dim=1600

#### training
bsz=16
num_workers=8
n_epoch=300

list="2025 2024 2023 2022 2021"

for seed in $list
do
  echo $seed

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PYTHONPATH:. python uvcom/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--eval_bsz 32 \
--results_root ${results_root} \
--exp_id org_layer4_query40_${lr} \
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--num_queries ${query_num} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--seed ${seed} \
--lr_gamma ${lr_gamma} \
--dec_layers ${dec_layers} \
--lr_drop ${lr_drop} \
--em_iter 5 \
--n_txt_mu ${n_txt_mu} \
--n_visual_mu ${n_visual_mu} \
--neg_choose_epoch ${neg_choose_epoch} \
--n_epoch ${n_epoch} \
--clip_length 8 \
--max_v_l 10000 \
--max_q_l 500 \
--num_workers 8 \
--max_es_cnt 300 \
--enc_layers 4 \
--dec_layers 4 \
--num_queries 40 \
${@:1}

done