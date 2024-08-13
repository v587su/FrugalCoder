device=$1
model=$2
lang=$3

if [ $lang == "python" ]; then
    dataset=python/final/jsonl/
else
    dataset=COFIC.jsonl
fi
CUDA_VISIBLE_DEVICES=$device python3 generate_score.py --exp_name tosem --device cuda:0 --mode score --model $model --batch_size 1 --text_length 256 --min_query_len 10 --language $lang --data_dir ./data --dataset_name $dataset