#!/bin/bash
#SBATCH --job-name=llava_v_1_6 # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=32 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:8 # number of gpus per node
#SBATCH -p pot

#SBATCH -o ./%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ./%x-%j.err

export CUDA_LAUNCH_BLOCKING=1 

deepspeed /cognitive_comp/sunrenliang/LLaVA/llava/train/train_mem.py \
    --deepspeed /cognitive_comp/sunrenliang/LLaVA/scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /cognitive_comp/sunrenliang/LLaVA/scripts/v1_5/playground/data/clean_sharegpt4v_instruct.json \
    --image_folder /cognitive_comp/sunrenliang/LLaVA/scripts/v1_5/playground/data/ \
    --vision_tower /cognitive_comp/sunrenliang/ECCV2024/LLaVA/scripts/v1_5/checkpoints/moe_clip_4 \
    --pretrain_mm_mlp_adapter /cognitive_comp/sunrenliang/LLaVA/scripts/v1_5/checkpoints/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /cognitive_comp/sunrenliang/LLaVA/scripts/v1_5/checkpoints/moe_llava_llava_finetune_data_for_test_moe_4_for_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
