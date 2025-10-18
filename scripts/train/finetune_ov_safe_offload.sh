export OMP_NUM_THREADS=8

# 内存安全的环境变量设置
export NCCL_IB_DISABLE=1              
export NCCL_P2P_DISABLE=1             
export NCCL_SOCKET_IFNAME=lo          
export NCCL_DEBUG=WARN
export NCCL_BUFFSIZE=1048576  # 1MB buffer
export NCCL_NTHREADS=2

# CUDA内存管理
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# 保守的内存配置
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0

# 清理GPU内存
export CUDA_VISIBLE_DEVICES=0

# 使用gcc wrapper
export CC=/home/zhou/LLaVA-NeXT/gcc_wrapper.sh
export CXX=/usr/bin/g++

# Triton环境变量
export TRITON_CACHE_DIR=/tmp/triton_cache
export TRITON_PRINT_AUTOTUNING=0

# 修复libstdc++版本问题
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD

# DeepSpeed优化
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0

export NUM_GPUS=1           
export NNODES=1            
export RANK=0              
export ADDR=localhost       
export PORT=29500     

LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# 非常保守的训练配置
Batchsize=1
Accumulation_steps=2  # 进一步减少
echo "Batch size: ${Batchsize}, Accumulation steps: ${Accumulation_steps}"

############### Finetune ################

PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_stage_am9_safe_offload" 
PREV_STAGE_CHECKPOINT="./checkpoints/onevision/llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-si_stage_am9"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${RUN_NAME}"

# 清理GPU内存
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# 启动内存监控
python monitor_memory.py &
MONITOR_PID=$!

# 设置清理函数
cleanup() {
    echo "清理资源..."
    kill $MONITOR_PID 2>/dev/null || true
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}
trap cleanup EXIT

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
    --image_aspect_ratio anyres_max_1 \
    --image_grid_pinpoints  "[[384,384],[384,768],[768,384]]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./checkpoints/onevision/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${Batchsize} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${Accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile False \
    --dataloader_drop_last True \
    --frames_upbound 2 \
    --attn_implementation flash_attention_2

exit 0;