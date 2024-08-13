device=$1
model=$2
estimator=$3
language=$4

for metric in bleu cbleu; do
    CUDA_VISIBLE_DEVICES=$device python3 train_estimator.py \
        --exp_name tosem \
        --device cuda:0 \
        --model $model \
        --language $language \
        --data_dir ./data/ \
        --metric $metric \
        --estimator $estimator \
        --text_length 256 \
        --output_dir ./output/trained_models/ \
        --epoch 10
done