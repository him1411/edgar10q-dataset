#!/bin/bash
set -x



export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export WANDB_DISABLED="true"

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port run_model.py \
    --model_name_or_path "allenai/tk-instruct-large-def-pos" \
    --cache_dir "/data/data/hgupta35/hf_cache" \
    --do_train true --do_eval true --do_predict true \
    --train_file "train.csv" \
    --validation_file "val.csv" \
    --test_file "test.csv" \
    --output_dir /output/ \
    --per_device_train_batch_size="8" --per_device_eval_batch_size="8" --gradient_accumulation_steps="8" \
    --max_source_length 512  --max_target_length 128 --generation_max_length 128 \
    --learning_rate 5e-05 --num_train_epochs 2   --warmup_steps 100 --predict_with_generate  --save_strategy=no \
    --overwrite_output_dir \
    --fsdp "full_shard auto_wrap" \
    --bf16 
    --run_name t5-experiment
    --deepspeed stage3_config.json \