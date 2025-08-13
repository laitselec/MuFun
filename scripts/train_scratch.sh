WARMUP_DATA_PATH=adata/warmup_data.json  
TRAINFULL_DATA_PATH=adata/trainfull_data.json   
WARMUP_EVAL_PATH=adata/eval_data.json  
TRAINFULL_EVAL_PATH=adata/eval_data.json 
WARMUP_AUDIO_PATH=/ # audio root dir, / for absolute path
TRAINFULL_AUDIO_PATH=/ 

LLM_VERSION=Qwen/Qwen3-0.6B-Base # llm path in huggingface
AT_VERSION=openai/whisper-large-v3 # audio tower path in huggingface
AT_VERSION2="" 
CN_VERSION=blp_4i_2x # connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_instruct # chat template, other options are: phi, llama, gemmma, etc
VERSION=exp1 # experiment name for recording different runnings
TRAIN_RECIPE=common # training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=768 # max model length for llm


# bash scripts/warmup_qwen.sh "$WARMUP_DATA_PATH" "$WARMUP_AUDIO_PATH" "$LLM_VERSION" "$AT_VERSION" "$AT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$WARMUP_EVAL_PATH"
bash scripts/trainfull_qwen.sh "$TRAINFULL_DATA_PATH" "$TRAINFULL_AUDIO_PATH" "$LLM_VERSION" "$AT_VERSION" "$AT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$TRAINFULL_EVAL_PATH"
