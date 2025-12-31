ray stop
env WANDB_API_KEY=75b560b94c4e949455fa45b3a018987ede3846fa RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0
python train.py
