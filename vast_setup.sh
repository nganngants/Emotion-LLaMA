touch ~/.no_auto_tmux

conda env create -f environment.yml

conda init

echo "conda activate llama" >> ~/.bashrc
source ~/.bashrc

apt-get update && apt-get install vim wget -y

apt-get install ffmpeg libsm6 libxext6 git-lfs -y

pip install gdown
pip install moviepy==1.0.3
pip install soundfile==0.12.1
pip install opencv-python==4.7.0.72

git clone https://huggingface.co/facebook/hubert-large-ll60k checkpoints/transformer/hubert-large-ll60k
gdown 1pNngqXdc3cKr9uLNW-Hu3SKvOpjzfzGY -O checkpoints/save_checkpoint/
gdown 1Vi_E7ZtZXRAQcyz4f8E6LtLh2UXABCmu -O checkpoints/minigptv2_checkpoint.pth 

mkdir mesc_features
cd mesc_features
gdown 1IM_Ac55pzLeqrvw4eamki9_JkkBILQ4O
gdown 11sS_E08AvQSL98nMzCaFhc7c-FGlfByT
gdown 1EItnR-mBGCvmvJCYKN8pZe0etWS71_bu

tar -xzf mesc_mae_features.tar.gz 
tar -xzf mesc_maevideo_features.tar.gz 
tar -xzf mesc_hubert_features.tar.gz
