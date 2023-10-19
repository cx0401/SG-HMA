set -o errexit

dataset=$1
model_type=$2
is_table=$3
export CUDA_VISIBLE_DEVICES=$4

seed=10
origin=rewrite0
out_rewrite=${model_type}_table${is_table}_rewrite0

# 第一轮训练
python train.py \
    --dataset $dataset \
    --train_src data/${dataset}/data/${origin}/train/src.txt \
    --train_gold data/${dataset}/data/${origin}/train/gold.txt \
    --train_table data/${dataset}/data/${origin}/train/table.json \
    --eval_src data/${dataset}/data/${origin}/test/src.txt \
    --eval_gold data/${dataset}/data/${origin}/test/gold.txt \
    --eval_table data/${dataset}/data/${origin}/test/table.json \
    --cand_train_path "" \
    --cand_train_metrics_path "" \
    --cand_eval_path "" \
    --cand_eval_metrics_path "" \
    --model_type ${model_type} \
    --output_dir data/${dataset}/checkpoint/${out_rewrite}  \
    --model_name_or_path "" \
    --rewrite "no" \
    --table $is_table \
    --mask "yes" \
    --seed $seed 

# 生成输出结果，同时准备用于下一轮的重写
for mode in "test" "eval" "train"
do
python run_generation.py \
    --dataset $dataset \
    --model_name_or_path data/${dataset}/checkpoint/${out_rewrite}  \
    --table $is_table \
    --mask "yes" \
    --curr_dir data/${dataset}/result/${out_rewrite}/${mode}/out.txt \
    --src_dir data/${dataset}/data/${origin}/${mode}/src.txt \
    --table_path data/${dataset}/data/${origin}/${mode}/table.json \
    --model_type $model_type \
    --gen_can "no" 

python cal_metrics.py \
    --src data/${dataset}/data/${origin}/${mode}/src.txt \
    --ref data/${dataset}/data/${origin}/${mode}/gold.txt \
    --hyp data/${dataset}/result/${out_rewrite}/${mode}/out.txt \
    --table_path data/${dataset}/result/${out_rewrite}/${mode}/table.json \
    --score "no" \
    --out_log_path data/${dataset}/metrics/${out_rewrite}/${mode}/metrics.json \
    --out_every_path data/${dataset}/metrics/${out_rewrite}/${mode}/every.json \
    --model_name_or_path  data/${dataset}/checkpoint/${out_rewrite}  \
    --model_type "gpt2" \
    -p -l
done

# 生成对比学习的负样本
for mode in "test" "eval" "train"
do
python run_generation.py \
    --dataset $dataset \
    --model_name_or_path data/${dataset}/checkpoint/${out_rewrite} \
    --table_path data/${dataset}/data/${origin}/${mode}/table.json \
    --table $is_table \
    --mask "yes" \
    --curr_dir data/${dataset}/result/${out_rewrite}/${mode}/sample.txt \
    --src_dir data/${dataset}/data/${origin}/${mode}/src.txt \
    --model_type $model_type \
    --gen_can "yes" \
    --num_beams 4 \
    --num_return_sequences 4

# 对负样本进行评分
python cal_metrics.py \
    --src data/${dataset}/data/${origin}/${mode}/src.txt \
    --ref data/${dataset}/data/${origin}/${mode}/gold.txt \
    --hyp data/${dataset}/result/${out_rewrite}/${mode}/sample.txt \
    --table_path data/${dataset}/data/${origin}/${mode}/table.json \
    --score "no" \
    --out_every_path data/${dataset}/metrics/${out_rewrite}/${mode}/sample.json \
    --model_name_or_path  data/${dataset}/checkpoint/${out_rewrite} \
    --model_type "gpt2" \
    -p -l
done

new_rewrite=${model_type}_table${is_table}_rewrite1
for mode in "test" "eval" "train"
do
python rewrite.py \
    --origin_data_dir data/${dataset}/data/${origin}/${mode}/src.txt \
    --out_data_dir data/${dataset}/result/${out_rewrite}/${mode}/out.txt \
    --rewrite_data_dir data/${dataset}/data/${new_rewrite}/${mode}/src.txt 
done

python train.py \
    --dataset $dataset \
    --train_src data/${dataset}/data/${new_rewrite}/train/src.txt \
    --train_gold data/${dataset}/data/${origin}/train/gold.txt \
    --train_table data/${dataset}/data/${origin}/train/table.json \
    --eval_src data/${dataset}/data/${new_rewrite}/eval/src.txt \
    --eval_gold data/${dataset}/data/${origin}/eval/gold.txt \
    --eval_table data/${dataset}/data/${origin}/eval/table.json \
    --cand_train_path data/${dataset}/result/${out_rewrite}/train/sample.txt \
    --cand_train_metrics_path data/${dataset}/metrics/${out_rewrite}/train/sample.json \
    --cand_eval_path data/${dataset}/result/${out_rewrite}/eval/sample.txt \
    --cand_eval_metrics_path data/${dataset}/metrics/${out_rewrite}/eval/sample.json \
    --model_type ${model_type} \
    --output_dir data/${dataset}/checkpoint/${new_rewrite} \
    --model_name_or_path data/${dataset}/checkpoint/${out_rewrite} \
    --rewrite "yes" \
    --table $is_table \
    --mask "yes"

mode="test"
python run_generation.py \
    --dataset $dataset \
    --model_name_or_path data/${dataset}/checkpoint/${new_rewrite} \
    --table_path data/${dataset}/data/${origin}/${mode}/table.json \
    --table $is_table \
    --mask "yes" \
    --curr_dir data/${dataset}/result/${new_rewrite}/${mode}/out.txt \
    --src_dir data/${dataset}/data/${new_rewrite}/${mode}/src.txt \
    --model_type $model_type \
    --gen_can "no" 

python cal_metrics.py \
    --src data/${dataset}/data/${origin}/${mode}/src.txt \
    --ref data/${dataset}/data/${origin}/${mode}/gold.txt \
    --hyp data/${dataset}/result/${new_rewrite}/${mode}/out.txt \
    --table_path data/${dataset}/data/${new_rewrite}/${mode}/table.json \
    --score "no" \
    --out_log_path data/${dataset}/metrics/${new_rewrite}/${mode}/metrics.json \
    --out_every_path data/${dataset}/metrics/${new_rewrite}/${mode}/every.json \
    --model_name_or_path  data/${dataset}/checkpoint/${new_rewrite} \
    --model_type "gpt2" \
    -p -l