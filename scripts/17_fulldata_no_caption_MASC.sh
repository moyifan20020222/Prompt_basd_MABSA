for seed in 42
do
    for sl in '6e-5'
    do
        for num_image_tokens in 2 4
        do
            for Prompt_Pool_num in 6 9 12
            do
            CUDA_VISIBLE_DEVICES=2 python twitter_sc_training_for_generated_prompt.py \
                    --dataset twitter17 src/data/jsons/few_shot_for_prompt/twitter_2017/twitter17_full_data_no_caption.json \
                    --checkpoint_dir ./ \
                    --model_config data/bart-base/config.json \
                    --log_dir log_for_generated_prompt/17_full_data_no_caption_${sl}_${num_image_tokens}_${Prompt_Pool_num}_sc \
                    --num_beams 4 \
                    --eval_every 1 \
                    --lr ${sl} \
                    --batch_size 16 \
                    --epochs 100 \
                    --grad_clip 5 \
                    --warmup 0.1 \
                    --is_sample 0 \
                    --seed ${seed} \
                    --num_image_tokens ${num_image_tokens} \
                    --task twitter_sc \
                    --num_workers 8 \
                    --has_prompt \
                    --use_caption False \
                    --use_generated_prompt \
                    --use_different_senti_prompt \
                    --Prompt_Pool_num ${Prompt_Pool_num}
            done
        done
    done
done
