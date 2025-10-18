export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

# training on single local computer
export NCCL_IB_DISABLE=1              
export NCCL_P2P_DISABLE=0             
export NCCL_SOCKET_IFNAME=lo          
export NCCL_DEBUG=INFO

# CUDA library path - fix Triton compilation issue
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Use gcc wrapper to add library paths for Triton compilation
export CC=/home/zhou/LLaVA-NeXT/gcc_wrapper.sh
export CXX=/usr/bin/g++

# Triton environment variables to avoid compilation issues
export TRITON_CACHE_DIR=/tmp/triton_cache
export TRITON_PRINT_AUTOTUNING=0

# Fix libstdc++ version issue for DeepSpeed CPU offload
# Use system libstdc++ instead of conda's outdated version
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #,max_split_size_mb:1024

export NUM_GPUS=1           
export NNODES=1            
export RANK=0              
export ADDR=localhost       
export PORT=29500     

LLM_VERSION="Qwen/Qwen2-0.5B-Instruct" #Qwen/Qwen2-0.5B-Instruct or Qwen/Qwen2-7B-Instruct
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Training config
Batchsize=1                              # Sub-batchsize for accumulation
Accumulation_steps=$((8 / Batchsize)) # original Accumulation steps = 512 / Batchsize
echo "Batch size: ${Batchsize}, Accumulation steps: ${Accumulation_steps}"

############### Pretrain ################

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-0.5B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_stage_am9" 
PREV_STAGE_CHECKPOINT="./checkpoints/onevision/llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-si_stage_am9" # replace it with your last checkpoint training from single image collection
#PREV_STAGE_CHECKPOINT="./checkpoints/onevision/llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mid_stage_am4_lora"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

# Note: mm_projector will be loaded from PREV_STAGE_CHECKPOINT (SI stage)
# No need to specify pretrain_mm_mlp_adapter as it would overwrite SI stage training

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3_offload_fixed.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path ./scripts/train/nano_onevision.yaml \
    --image_folder ./data/images \
    --video_folder ./data/videos \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_0 \
    --image_grid_pinpoints  "[[384,384]]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./checkpoints/onevision/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${Batchsize} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${Accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 6144 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile False \
    --dataloader_drop_last True \
    --frames_upbound 7 \
    --attn_implementation flash_attention_2 \
# --lora_enable True \
# --lora_r 4 \
# --lora_alpha 8 \
# --lora_dropout 0.0 \
# --lora_bias "none" \
# You can delete the sdpa attn_implementation if you want to use flash attn
#    --mm_vision_tower_lr=2e-6 \
#    --frames_upbound 32 \
#    The orginal resolution is --image_grid_pinpoints  "(1x1),...,(2x2)" \ [[384,384],[384,768],[768,384]]
exit 0;

