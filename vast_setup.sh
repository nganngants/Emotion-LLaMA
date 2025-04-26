touch ~/.no_auto_tmux

conda install -c nvidia cuda-compiler

conda env create -f environment.yml

conda init

echo "conda activate llama" >> ~/.bashrc
source ~/.bashrc

apt-get update && apt-get install vim wget -y

pip install -U gdown
gdown 1qobIrZ_3vfDzWSU81MN4Hs1_iD1xhAIm
gdown 1KZEd-KrOsF8BzahlP-v3pGKlR1xvRcD2