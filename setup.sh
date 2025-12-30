cd ~
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
# 先删除旧的符号链接（如果有的话）
sudo rm -rf /usr/local/cuda

# 建立新的符号链接
sudo ln -s /usr/local/cuda-12.8 /usr/local/cuda

# 读总内存（单位：kB）
mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)

# 512GB = 512 * 1024 * 1024 kB
limit_kb=$((512 * 1024 * 1024))

if [ "$mem_kb" -lt "$limit_kb" ]; then
    export MAX_JOBS=8
    echo "内存低于 512GB，已设置 MAX_JOBS=8"
else
    echo "内存大于等于 512GB，不修改 MAX_JOBS"
fi

cd ~
conda tos accept --override-channels --channel  https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel  https://repo.anaconda.com/pkgs/r
conda create -y -n agl python=3.10
conda init
source ~/.bashrc
conda activate agl
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install verl==0.6.0 transformers==4.57.1 vllm==0.10.2
git clone https://github.com/Dao-AILab/flash-attention.git
cd ~/flash-attention
git reset --hard 0e60e39473e8df549a20fb5353760f7a65b30e2d
pip install packaging
pip install ninja
python setup.py install
pip install datasets
pip install click==8.2.1
pip install cachetools==5.5.2
pip install ray==2.49.2

pip install tensordict==0.6.2 torchdata==0.11.0 agentops==0.4.14
pip install "autogen-agentchat" "autogen-ext[openai]"
pip install flask mcp
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

conda activate agl

cd ~/agent-lightning/
pip install -e .