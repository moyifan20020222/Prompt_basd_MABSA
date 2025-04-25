for seed in 13
do
    for sl in  '5e-5'
    do
        for num_image_tokens in 2 4
        do
            for Prompt_Pool_num in 8 10
            do
            CUDA_VISIBLE_DEVICES=0 python twitter_sc_training_for_generated_prompt.py \
                    --dataset twitter15 src/data/jsons/few_shot_for_prompt/twitter_2015/twitter15_full_data.json \
                    --checkpoint_dir ./ \
                    --model_config data/bart-base/config.json \
                    --log_dir log_for_generated_prompt/15_full_data_${seed}_${sl}_${num_image_tokens}_${Prompt_Pool_num}_sc \
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
                    --use_caption True \
                    --use_generated_prompt \
                    --use_different_senti_prompt \
                    --Prompt_Pool_num ${Prompt_Pool_num}
            done
        done
    done
done
