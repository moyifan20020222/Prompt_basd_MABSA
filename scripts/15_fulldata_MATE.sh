for loss_lambda in 0.1
do
    for sl in  '6.5e-5'
    do
        for seed in 13
        do
            for num_image_tokens in 4
            do
                for Prompt_pool_num in 8 12
                do
                CUDA_VISIBLE_DEVICES=1 python twitter_ae_training_for_generated_prompt_multitasks.py \
                        --dataset twitter15 src/data/jsons/few_shot_for_prompt/twitter_2015/twitter15_full_data.json \
                        --checkpoint_dir ./ \
                        --model_config data/bart-base/config.json \
                        --log_dir log_for_generated_aspect_prompt_multitasks/15_full_data_${sl}_${num_image_tokens}_${Prompt_pool_num}_ae \
                        --num_beams 4 \
                        --eval_every 1 \
                        --lr ${sl} \
                        --batch_size 4 \
                        --epochs 100 \
                        --grad_clip 5 \
                        --warmup 0.1 \
                        --is_sample 0 \
                        --seed ${seed} \
                        --task twitter_ae \
                        --num_workers 8 \
                        --num_image_tokens ${num_image_tokens} \
                        --loss_lambda ${loss_lambda} \
                        --use_multitasks \
                        --has_prompt \
                        --use_generated_prompt \
                        --use_different_aspect_prompt \
                        --Prompt_Pool_num ${Prompt_pool_num} \
                        --use_caption True
                done
            done
        done
    done
done