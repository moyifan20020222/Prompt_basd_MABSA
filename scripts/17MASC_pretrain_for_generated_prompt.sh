for seed in 42 87 100
do
    for sl in  '7.5e-5' 
    do
        for num_image_tokens in 2 3 4
        do
        CUDA_VISIBLE_DEVICES=3 python twitter_sc_training_for_generated_prompt.py \
                --dataset twitter17 src/data/jsons/few_shot_for_prompt/twitter_2017/twitter17_${seed}_info.json \
                --checkpoint_dir ./ \
                --model_config data/bart-base/config.json \
                --log_dir log_for_generated_prompt/17_${seed}_${sl}_${num_image_tokens}_sc \
                --num_beams 4 \
                --eval_every 1 \
                --lr ${sl} \
                --batch_size 16  \
                --epochs 100 \
                --grad_clip 5 \
                --warmup 0.1 \
                --is_sample 0 \
                --seed ${seed} \
                --num_image_tokens ${num_image_tokens} \
                --task twitter_sc \
                --num_workers 8 \
                --has_prompt \
                --use_caption \
                --use_generated_prompt \
                --use_different_senti_prompt \
                --Prompt_Pool_num 12
        done
    done
done
