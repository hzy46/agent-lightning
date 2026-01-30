ray stop

export NCCL_CUMEM_HOST_ENABLE=0

env NCCL_CUMEM_HOST_ENABLE=0 \
    WANDB_API_KEY=75b560b94c4e949455fa45b3a018987ede3846fa \
    RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0

python data/gen_vt.py --min_answer_num 1 --max_answer_num 4

python train.py --method stream --train_gsm_lengths "[]" \
    --train_kv_lengths "['16K']" \
    --from_model Qwen/Qwen2.5-7B-Instruct \
    --agg_mode True \
    --train_kv_subset minans1_maxans4 \
    --fix_chunk_num 4 \
    --max_rounds 6
