import glob
import os
import json
import pickle
import random
import time
import itertools
import pandas as pd
import json
from typing import List
import torch.nn.functional as F

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class MESCData:
    label_utt: str
    label_emotion: str
    label_strategy: str
    video_path: str
    user_utt: str
    user_emotion: str

    def __init__(self, label_utt, label_emotion, label_strategy, video_path, user_utt, user_emotion):
        self.label_utt = label_utt
        self.label_emotion = label_emotion
        self.label_strategy = label_strategy
        self.video_path = video_path
        self.user_utt = user_utt
        self.user_emotion = user_emotion

class MESCDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, video_dir, jsonl_path, features_dir):
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.system_instruction_prefix = "You are a 'Therapist' analyzing a video session. Based on what client says and client's emotion, "
        self.emotion_instruction_pool = [
            "Please determine which emotion label in the video represents: anger, sadness, disgust, fear, depression, neutral, joy.",
        ]

        self.system_emotion_instruction_pool = [
            "Please determine which emotion you should express to the client in the video: anger, sadness, disgust, fear, depression, neutral, joy.",
        ]
        
        self.system_strategy_instruction_pool = [
            "Please determine which strategy you should use to respond to the client in the video: open question, approval, self-disclosure, restatement, interpretation, advisement, communication skills, structuring the therapy, guiding the pace and depth of the conversation, others.",
        ]

        self.system_answer_instruction_pool = [
            "Please generate a response to the client in the video, based on the emotion you should express and the strategy you should use. Your reply should:\n"
            "- Understand and acknowledge the client's emotion and perspective.\n"
            "- Express sympathy for negative situations or approval for positive ones.\n"
            "- Avoid negative triggers (disgust, resentment, discrimination, hatred, etc.).\n"
            "- Be truthful, supportive, and foster understanding and comfort.\n"
            "- Repeat a few words from the client's utterance.\n"
            "- Express a different opinion if needed, but never hurt the client's feelings.\n"
            "- Safeguard human autonomy, identity, and data dignity.\n"
        ]

        # self.task_pool = [
        #    "emotion",
        #    "reason",
        #    "infer",
        # ]

        self.task_pool = [
           "emotion",
           "system_emotion",
            "system_strategy",
            "system_answer"
        ]

        self.jsonl_path = jsonl_path
        self.features_dir = features_dir
        self.video_dir = video_dir
        
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(MESCData(
                    label_utt=" ".join(item['Utterance']),
                    label_emotion=item['Emotion'],
                    label_strategy=item['Strategy'],
                    video_path=os.path.join(self.video_dir, item['path_to_vid_user_most_recent'][-1]),
                    user_utt=". ".join(item['utt_user_most_recent']),
                    user_emotion=item['get_emotion_user_most_recent']
                ))

        emos = ['anger', 'sadness', 'disgust', 'fear', 'depression', 'neutral', 'joy']

        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(emos): self.emo2idx[emo] = ii
        for ii, emo in enumerate(emos): self.idx2emo[ii] = emo

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        image = self.extract_frame(item.video_path)
        video_name = item.video_path.split('/')[-1].split('.')[0]

        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)

        FaceMAE_feats, VideoMAE_feats, Audio_feats = self.get(video_name)
        if len(VideoMAE_feats.shape) == 1:
            VideoMAE_feats = VideoMAE_feats.unsqueeze(0)
        if len(Audio_feats.shape) == 1:
            Audio_feats = Audio_feats.unsqueeze(0)
        if len(FaceMAE_feats.shape) == 1:
            FaceMAE_feats = FaceMAE_feats.unsqueeze(0)
        video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)

        task = random.choice(self.task_pool)
        if task == "emotion":
            caption = item.user_utt # llama2 putput only emotion class
            caption = self.text_processor(caption)
            instruction_pool = self.emotion_instruction_pool
        elif task == "system_emotion":
            caption = item.label_emotion
            caption = self.text_processor(caption)
            instruction_pool = self.system_emotion_instruction_pool
        elif task == "system_strategy":
            caption = item.label_strategy
            caption = self.text_processor(caption)
            instruction_pool = self.system_strategy_instruction_pool
        elif task == "system_answer":
            caption = item.label_utt
            caption = self.text_processor(caption)
            instruction_pool = self.system_answer_instruction_pool

        emotion = self.emo2idx[item.user_emotion]
        character_line = "The person in video says: {}. ".format(item.user_utt)

        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        return {
            "image": image,
            "video_features": video_features,
            "instruction_input": instruction,
            "answer": caption,
            "emotion": emotion,
            "image_id": item.video_path.split('/')[-1].split('.')[0]
        }

    def extract_frame(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        success, frame = video_capture.read()
        if not success:
            raise ValueError("Failed to read video file:", video_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_capture.release()

        return frame_rgb
    
    def get(self, video_name):
        FaceMAE_feats = np.load(os.path.join(self.features_dir, 'mesc_mae_features', video_name + '.mp4.npy'))
        VideoMAE_feats = np.load(os.path.join(self.features_dir, 'mesc_maevideo_features', video_name + '.mp4.npy'))
        Audio_feats = np.load(os.path.join(self.features_dir, 'mesc_hubert_features', video_name + '.npy'))

        FaceMAE_feats = torch.tensor(FaceMAE_feats)
        VideoMAE_feats = torch.tensor(VideoMAE_feats)
        Audio_feats = torch.tensor(Audio_feats)

        return FaceMAE_feats, VideoMAE_feats, Audio_feats