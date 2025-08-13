TRAIN_DATA_PATH="adata/train_data.json"
EVAL_DATA_PATH="adata/eval_data.json"
AUDIO_PATH="/"
LLM_VERSION="Qwen/Qwen3-8B-Base"
AT_VERSION="openai/whisper-large-v3"
AT_VERSION2=""
CN_VERSION="blp_4i_2x"
CONV_VERSION="qwen2_instruct"
VERSION="exp1"
TRAIN_RECIPE="common"
MODEL_MAX_LENGTH=4096


deepspeed --include localhost:0,1,2,3,4,5 --master_port 29501 tinyllava/train/train.py \
    --deepspeed ./scripts/z3.json \
    --data_path  $TRAIN_DATA_PATH \
    --eval_data_path  $EVAL_DATA_PATH \
    --image_folder $AUDIO_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $AT_VERSION \
    --vision_tower2 "$AT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio audio \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower full \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length False \
    --pretrained_model_path Yi3852/MuFun-Base \
    --output_dir checkpoints/${VERSION}-ft1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size  2 \
    --gradient_accumulation_steps 12 \
    --eval_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --tokenizer_use_fast False \
    --run_name mufun-${VERSION}-ft1 
