set -x
source ~/.bashrc
source ~/azureml_job_env.sh
conda activate vllm
"$@"
