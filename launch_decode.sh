#export NVSHMEM_ENABLE_NIC_PE_MAPPING=1 
export NVSHMEM_IB_GID_INDEX=3
# HCA 列表按 GPU NUMA 亲和性排列: GPU0->mlx5_3, GPU1->mlx5_2, ...
export NVSHMEM_HCA_LIST=mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1

#export NVSHMEM_DEBUG=INFO
#export NVSHMEM_DEBUG_SUBSYS=INIT,TRANSPORT

export DEVICE_NAMES=mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1
export LOCAL_IP=10.6.131.21
export DECODE_MASTER_IP=10.6.131.20
export NCCL_TIMEOUT=6000
export SGLANG_DG_CACHE_DIR="./dgcache/1p1dcache/cache"
export SGLANG_TBO_DEBUG=1
export SGLANG_HACK_DEEPEP_NEW_MODE=0
export SGLANG_NUM_RESERVED_DECODE_TOKENS=256
export SGLANG_HACK_NUM_QPS_PER_RANK_ALI=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export PYTHONUNBUFFERED=1
export SGLANG_RECORD_STEP_TIME=1

#export NVSHMEM_DEBUG=INFO
#export NVSHMEM_DEBUG_SUBSYS=INIT,TRANSPORT

export SGLANG_HEALTH_CHECK_TIMEOUT=240
python -m sglang.launch_server \
    --model-path /models/DeepSeek-R1 \
    --disaggregation-ib-device ${DEVICE_NAMES} \
    --disaggregation-mode decode \
    --host ${LOCAL_IP} \
    --port 30001 \
    --trust-remote-code \
    --dist-init-addr ${DECODE_MASTER_IP}:5000 \
    --nnodes 2 \
    --node-rank 1 \
    --tp-size 16 \
    --dp-size 16 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --mem-fraction-static 0.8 \
    --decode-log-interval 1 \
    --disaggregation-transfer-backend nixl \
    --max-running-requests 208 \
    --deepep-mode low_latency \
    --cuda-graph-bs 13 \
    --enable-dp-lm-head \
    --moe-dense-tp-size 1 \
    --attention-backend flashinfer \
    --watchdog-timeout 6000 \
    --dist-timeout 6000 \
    --prefill-round-robin-balance \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 2 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 3
