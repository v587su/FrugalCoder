
for model in starcoder codegen
do
    for metric in cbleu bleu
    do
        for estimator in tcqe bert lstm ada lr
        do
            for language in python java
            do
                CUDA_VISIBLE_DEVICES=$1 python3 eval_estimator.py \
                    --exp_name tosem \
                    --device cuda:0 \
                    --model $model \
                    --language $language \
                    --data_dir ./data \
                    --metric $metric \
                    --estimator $estimator \
                    --output_dir ./output/trained_models \
                    --load_prediction
            done
        done
    done
done