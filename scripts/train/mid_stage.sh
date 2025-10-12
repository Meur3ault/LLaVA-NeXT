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

# Fix libstdc++ version issue for DeepSpeed CPU offload
# Use system libstdc++ instead of conda's outdated version
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD

# Triton environment variables to avoid compilation issues
export TRITON_CACHE_DIR=/tmp/triton_cache
export TRITON_PRINT_AUTOTUNING=0

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
Batchsize=1                              # Sub-batchsize for accumulation (reduced for single GPU)
Accumulation_steps=$(((256 / 2) / Batchsize)) # Accumulation steps = 512 / Batchsize
echo "Batch size: ${Batchsize}, Accumulation steps: ${Accumulation_steps}"

############### Pretrain ################

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-0.5B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 1.5
PROMPT_VERSION="qwen_1_5"

# Pretrained projector
BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
PRETRAIN_PROJECTOR="./checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin"

# Mid Stage name
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mid_stage_am4"

echo "============================================"
echo "Mid Stage Training Configuration"
echo "============================================"
echo "LLM Model: ${LLM_VERSION}"
echo "Pretrained Projector: ${PRETRAIN_PROJECTOR}"
echo "Run Name: ${RUN_NAME}"
echo "============================================"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version $PROMPT_VERSION \
    --data_path ./scripts/train/nano_mid_stage.yaml \
    --image_folder ./data/images \
    --pretrain_mm_mlp_adapter ${PRETRAIN_PROJECTOR} \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --lora_enable True \
    --lora_r 4 \
    --lora_alpha 8 \
    --lora_dropout 0.0 \
    --lora_bias "none" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_2 \
    --image_grid_pinpoints  "[[384,384],[384,768],[768,384]]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${RUN_NAME} \
    --output_dir ./checkpoints/onevision/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${Batchsize} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${Accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --attn_implementation sdpa \

#     --model_max_length 32768 \
#    The orginal resolution is --image_grid_pinpoints  "[[768,768],[384,768],[384,1152],[768,384],[1152,384]]" \ 
# you could try to use flash attention instead of sdpa, but it needs to be installed manually.
#     --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" that is too big
exit 0;
