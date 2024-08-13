

CUDA_VISIBLE_DEVICES=$1 python3 eval_estimator.py \
    --exp_name tosem \
    --device cuda:0 \
    --model $2 \
    --language $3 \
    --data_dir ./data \
    --metric $4 \
    --estimator $5 \
    --output_dir ./output/trained_models 
