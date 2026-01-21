ray stop

export NCCL_CUMEM_HOST_ENABLE=0

env NCCL_CUMEM_HOST_ENABLE=0 \
    WANDB_API_KEY=75b560b94c4e949455fa45b3a018987ede3846fa \
    RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0

# env NCCL_IGNORE_DISABLED_P2P=1 \
    # NCCL_P2P_DISABLE=1 \
    # NCCL_CUMEM_HOST_ENABLE=0 \

python train.py --method parallel --train_gsm_lengths "['16K']" --fix_chunk_num 2 --use_new_gen_data True