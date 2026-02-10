#!/usr/bin/env bash

set -e

MODEL_PATH="$1"

if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <model_path>"
  exit 1
fi

# 检测 GPU 数量
if ! command -v nvidia-smi &> /dev/null; then
  echo "nvidia-smi not found. No NVIDIA GPU detected."
  exit 1
fi

GPU_COUNT=$(nvidia-smi -L | wc -l)

if [ "$GPU_COUNT" -eq 0 ]; then
  echo "No NVIDIA GPU detected."
  exit 1
fi

echo "Detected $GPU_COUNT GPU(s)"
echo "Model path: $MODEL_PATH"

BASE_PORT=8000

for ((i=0; i<GPU_COUNT; i++)); do
  PORT=$((BASE_PORT + i))

  echo "Starting vLLM server on GPU $i (port $PORT)..."

  CUDA_VISIBLE_DEVICES=$i \
  vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --host 0.0.0.0 \
    --port "$PORT" \
    > "vllm_gpu_${i}.log" 2>&1 &

done

echo "All vLLM shard servers started."
echo "Logs: vllm_gpu_<gpu_id>.log"
