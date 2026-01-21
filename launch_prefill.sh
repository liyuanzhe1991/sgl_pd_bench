
#SGLANG_HACK_NUM_QPS_PER_RANK_ALI=1 \
#export SGLANG_HACK_DEEPEP_NEW_MODE=0 
#export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_HCA_LIST=mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1


export MC_TE_METRIC=true 
export PYTHONUNBUFFERED=1 

export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
#export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
.
export DEVICE_NAMES=mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1 
export SGLANG_DG_CACHE_DIR="/mnt/launch/dgcache/1p1dcache"
export LOCAL_IP=10.6.131.23
export PREFILL_MASTER_IP=10.6.131.23

export SGLANG_LOG_FORWARD_ITERS=1
export SGLANG_HEALTH_CHECK_TIMEOUT=240
export SGLANG_TORCH_PROFILER_DIR="/mnt/launch/torch_profile_logs"
#export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=18000
#export SGL_JIT_DEEPGEMM_COMPILE_WORKERS=16

#export DEEP_EP_TIMEOUT=1800000

#export NVSHMEM_DEBUG=INFO
#export NVSHMEM_DEBUG_SUBSYS=INIT,TRANSPORT

python3 -m sglang.launch_server \
    --model-path /models/DeepSeek-R1 \
    --disaggregation-mode prefill \
    --skip-tokenizer-init \
    --disaggregation-ib-device ${DEVICE_NAMES} \
    --host ${LOCAL_IP} \
    --port 30500 \
    --trust-remote-code \
    --dist-init-addr ${PREFILL_MASTER_IP}:6040 \
    --nnodes 2 \
    --node-rank 0 \
    --tp-size 16 \
    --load-balance-method round_robin \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 8000 \
    --max-running-requests 512 \
    --context-length 10000 \
    --dist-timeout 18000000 \
    --enable-expert-distribution-metrics \
    --disable-radix-cache \
    --watchdog-timeout 10000000 \
    --decode-log-interval 1 \
    --dp-size 16 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --moe-dense-tp-size 1 \
    --disaggregation-transfer-backend nixl \
    2>&1 | tee /mnt/launch/prefill.log


