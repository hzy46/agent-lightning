#!/usr/bin/env bash

# 1. 验证 ZHIYUHE 环境变量
if [[ -z "$ZHIYUHE" ]]; then
  echo "Error: ZHIYUHE is not set. Please export ZHIYUHE before running."
  exit 1
fi


# 2. 参数检查
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <exp_name> <steps_comma_separated>"
  echo "Example: $0 <exp_name> 50,100,150"
  exit 1
fi

exp_name="$1"
steps_csv="$2"

# 将逗号分隔的 steps 拆成数组
IFS=',' read -r -a steps <<< "$steps_csv"

# 3. 循环执行 merge
for step in "${steps[@]}"; do
  local_dir="parallel_agent/checkpoints/ParallelAgent/${exp_name}/global_step_${step}/actor/"
  target_dir="$ZHIYUHE/models/${exp_name}/global_step_${step}"
  mkdir -p $target_dir

  echo ">>> merging step ${step} ..."
  python model_merger.py merge \
    --backend fsdp \
    --local_dir "$local_dir" \
    --target_dir $target_dir
done

echo "All done ✅"