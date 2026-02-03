ray stop



# 1. 断言环境变量 ZHIYUHE 存在
if [[ -z "${ZHIYUHE:-}" ]]; then
  echo "Error: ZHIYUHE 环境变量不存在"
  exit 1
fi

SRC="$ZHIYUHE/train_qwen2.5-7b_stream_kv_v2_maxhop4_maxans2_16K_max_rounds_6_fix_chunk_num_4_agg/global_step_200"
DST="$HOME/train_qwen2.5-7b_stream_kv_v2_maxhop4_maxans2_16K_max_rounds_6_fix_chunk_num_4_agg/global_step_200"

# 2. 如果目标目录已存在，则不拷贝
if [[ -d "$DST" ]]; then
  echo "目标目录已存在，跳过拷贝：$DST"
  exit 0
fi

# 3. 创建 DST 的父目录（如果不存在）
mkdir -p "$(dirname "$DST")"

# 4. 拷贝目录
cp -r "$SRC" "$DST"

echo "拷贝完成：$SRC -> $DST"


export NCCL_CUMEM_HOST_ENABLE=0

env NCCL_CUMEM_HOST_ENABLE=0 \
    WANDB_API_KEY=75b560b94c4e949455fa45b3a018987ede3846fa \
    RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0

python data/gen_kv.py --max_hop_num 4 --max_answer_num 2

python train.py --method stream --train_gsm_lengths "[]" \
    --train_kv_lengths "['16K']" \
    --from_model Qwen/Qwen2.5-7B-Instruct \
    --agg_mode True \
    --train_kv_subset maxhop4_maxans2 \
    --fix_chunk_num 4 \
    --max_rounds 6 \
    --use_sparse_reward True \
    --max_sparse_reward 0.15 \
    --from_model $DST
