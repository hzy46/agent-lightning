#!/usr/bin/env bash

set -e

echo "Searching for vLLM serve processes..."

PIDS=$(ps -ef | grep "[v]llm serve" | awk '{print $2}')

if [ -z "$PIDS" ]; then
  echo "No vLLM serve processes found."
  exit 0
fi

echo "Found vLLM serve PIDs:"
echo "$PIDS"

for pid in $PIDS; do
  echo "Killing PID $pid"
  kill "$pid"
done

echo "All vLLM serve processes killed."
