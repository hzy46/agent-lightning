ray stop

export NCCL_CUMEM_HOST_ENABLE=0

env NCCL_CUMEM_HOST_ENABLE=0 \
    WANDB_API_KEY=75b560b94c4e949455fa45b3a018987ede3846fa \
    RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0

# env NCCL_IGNORE_DISABLED_P2P=1 \
    # NCCL_P2P_DISABLE=1 \
    # NCCL_CUMEM_HOST_ENABLE=0 \


# normal
# python train.py --method normal --train_gsm_lengths "[]" \
#     --train_kv_lengths "['8K']" \
#     --from_model Qwen/Qwen2.5-7B-Instruct


# # stream
# python train.py --method stream --train_gsm_lengths "[]" \
#     --fix_chunk_num 2 \
#     --train_kv_lengths "['16K']" \
#     --from_model ~/models/train_qwen2.5-7b_normal_kv_8K/global_step_400 \
#     --agg_mode True


python train.py --method stream --train_gsm_lengths "[]" \
    --fix_chunk_num 2 \
    --train_kv_lengths "['8K','16K']" \
    --from_model Qwen/Qwen2.5-7B-Instruct \
    --agg_mode True \
    --train_kv_subset maxhop2_maxans4
