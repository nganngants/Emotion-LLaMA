# Guide on finetuning Emotion-LLaMA on MESC dataset

## Note
Replace all the /workspace/Emotion-LLaMA in this repo with your own path.

## Environment
```
bash vast_setup.sh

# the following command require access token from HuggingFace (remember to install git-lfs first which already installed in vast_setup.sh)
rm -rf checkpoints/Llama-2-7b-chat-hf
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf checkpoints/Llama-2-7b-chat-hf
```

## Dataset
Download data and put it in the `data` folder:
```
data
|--video_data/
|--test.jsonl
|--train.jsonl
```

## Train config 
Edit the `train_configs/minigptv2_tuning_stage_2.yaml` file to set the training parameters. The key configurations are:

## Train
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 train.py --cfg-path train_configs/Emotion-LLaMA_finetune.yaml