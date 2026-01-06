conda tos accept --override-channels --channel  https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel  https://repo.anaconda.com/pkgs/r
conda create -y -n vllm python=3.10
conda init
source ~/.bashrc
conda activate vllm
pip install vllm==0.8.2
pip install "ray[serve]" fire

sudo apt update
sudo apt install aria2
bash hfd.sh BytedTsinghua-SIA/hotpotqa --dataset --tool aria2c -x 10 --local-dir ~/ruler_from_memagent

# start llm like
# python llm070.py --model Qwen/Qwen2.5-7B-Instruct-1M --tp 1
