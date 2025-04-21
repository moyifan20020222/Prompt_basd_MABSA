for seed in 13 42 100
do
    for sl in  '8e-5'
    do
        for num_image_tokens in 4
        do
        python twitter_sc_training_for_generated_prompt.py \
                --dataset twitter15 src/data/jsons/few_shot_for_prompt/twitter_2015/twitter15_${seed}_info.json \
                --checkpoint_dir ./ \
                --model_config data/bart-base/config.json \
                --log_dir log_for_generated_prompt/15_${seed}_sc \
                --num_beams 4 \
                --eval_every 1 \
                --lr ${sl} \
                --batch_size 4 \
                --epochs 100 \
                --grad_clip 5 \
                --warmup 0.1 \
                --is_sample 0 \
                --seed ${seed} \
                --num_image_tokens ${num_image_tokens} \
                --task twitter_sc \
                --num_workers 0 \
                --has_prompt \
                --use_caption \
                --use_generated_prompt \
                --use_different_senti_prompt \
                --cpu  # 新增强制使用CPU的参数
        done
    done
done