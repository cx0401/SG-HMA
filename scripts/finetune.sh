set -o errexit

dataset=wikibio
model_type=bart
is_table=no

lr=1e-5
seed=10
rewrite=${model_type}_table${is_table}_rewrite0
out_rewrite=${rewrite}

# 处理数据
# for mode in "test" "eval" "train"
# do
#     python data_process/${dataset}.py \
#         --model_type ${model_type} \
#         --source_path data/${dataset}/src1_${mode}.txt \
#         --gold_dir afs/${dataset}/data/${rewrite}/${mode}/gold.txt \
#         --src_dir afs/${dataset}/data/${rewrite}/${mode}/src.txt
# done

# 第一轮训练
python train.py \
    --dataset $dataset \
    --train_src afs/${dataset}/data/${rewrite}/train/src.txt \
    --train_gold afs/${dataset}/data/${rewrite}/train/gold.txt \
    --train_table afs/${dataset}/data/${rewrite}/train/table.json \
    --eval_src afs/${dataset}/data/${rewrite}/test/src.txt \
    --eval_gold afs/${dataset}/data/${rewrite}/test/gold.txt \
    --eval_table afs/${dataset}/data/${rewrite}/test/table.json \
    --cand_train_path "" \
    --cand_train_metrics_path "" \
    --cand_eval_path "" \
    --cand_eval_metrics_path "" \
    --model_type ${model_type} \
    --output_dir afs/${dataset}/checkpoint/${out_rewrite}  \
    --model_name_or_path "" \
    --rewrite "no" \
    --table $is_table \
    --mask "yes" \
    --learning_rate $lr \
    --seed $seed 

# 生成输出结果，同时准备用于下一轮的重写
for mode in "test" "eval" "train"
do
python run_generation.py \
    --dataset $dataset \
    --model_name_or_path afs/${dataset}/checkpoint/${out_rewrite}  \
    --table $is_table \
    --mask "yes" \
    --curr_dir afs/${dataset}/result/${out_rewrite}/${mode}/out.txt \
    --src_dir afs/${dataset}/data/${rewrite}/${mode}/src.txt \
    --table_path afs/${dataset}/data/${rewrite}/${mode}/table.json \
    --model_type $model_type \
    --gen_can "no" 

python cal_metrics.py \
    --src afs/${dataset}/data/${rewrite}/${mode}/src.txt \
    --ref afs/${dataset}/data/${rewrite}/${mode}/gold.txt \
    --hyp afs/${dataset}/result/${out_rewrite}/${mode}/out.txt \
    --score "no" \
    --out_log_path afs/${dataset}/metrics/${out_rewrite}/${mode}/metrics.json \
    --out_every_path afs/${dataset}/metrics/${out_rewrite}/${mode}/every.json \
    --model_name_or_path  afs/${dataset}/checkpoint/${out_rewrite}  \
    --model_type $model_type \
    -p -l
done
