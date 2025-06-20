touch ~/.no_auto_tmux

conda create -n demo python=3.10 -y

conda init

echo "conda activate demo" >> ~/.bashrc
source ~/.bashrc


apt-get update && apt-get install vim wget -y

apt-get install ffmpeg libsm6 libxext6 -y

sudo apt-get install git-lfs libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 -y
pip install -r requirements.txt
pip install moviepy==1.0.3
pip install soundfile==0.12.1
pip install opencv-python==4.7.0.72
pip install gradio==4.44.1
git config --global user.email "nganngants@gmail.com"
git config --global user.name "nganngants"

git clone https://huggingface.co/facebook/hubert-large-ll60k checkpoints/transformer/hubert-large-ll60k
gdown 11MmsYGVAIE3-T79DyIGahwN_eqL_6UNU -O checkpoints/save_checkpoint/Emotion-LLaMA.pth
# rm -rf checkpoints/Llama-2-7b-chat-hf
# git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf checkpoints/Llama-2-7b-chat-hf
# https://huggingface.co/facebook/hubert-large-ll60k
