from torch.utils.data import Dataset
import os
import pandas as pd
import json
from PIL import Image

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

class MESCDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor


        self.emotion_instruction_pool = [
            "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise.",
            # "Identify the displayed emotion in the video: is it happy, sad, neutral, angry, worried, or surprise?",
            # "Determine the emotional state shown in the video, choosing from happy, sad, neutral, angry, worried, or surprise.",
            # "Please ascertain the specific emotion portrayed in the video, whether it be happy, sad, neutral, angry, worried, or surprise.",
            # "Assess and label the emotion evident in the video: could it be happy, sad, neutral, angry, worried, surprise?",
        ]

        self.task_pool = [
           "emotion",
        ]

        print("ann_path: ", ann_path)
        self.ann_path = ann_path
        self.file_path = os.path.dirname(ann_path)
        self.tmp = read_jsonl(ann_path)
        print(('number of videos:%d' % (len(self.tmp))))

        # emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']
        self.emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise', 'fear', 'contempt', 'doubt']

        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(self.emos): self.emo2idx[emo] = ii
        for ii, emo in enumerate(self.emos): self.emo2idx[ii] = emo

        # # MER2024 transcription
        self.character_lines = pd.read_csv('/home/czb/big_space/datasets/Emotion/MER2024/transcription_all_new.csv')


    def load_data(self):
        # Load your dataset here
        # For example, you can use pandas to read a CSV file
        # df = pd.read_csv(os.path.join(self.data_dir, 'data.csv'))
        # return df
        pass

    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, index):
        t = self.tmp[index]
        video_name = t["path_to_vid_user_most_recent"][:-4] # remove the .mp4 part

        video_path = os.path.join(self.vis_root, video_name + ".mp4")
        if os.path.exists(video_path):
            image = self.extract_frame(video_path)
        else:
            raise ValueError("Video file not found:", video_path)

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

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            caption = t[2] # llama2 putput only emotion class
            caption = self.text_processor(caption)
            instruction_pool = self.emotion_instruction_pool
        
        emotion = self.emo2idx[t[2]]
        sentence = self.character_lines.loc[self.character_lines['name'] == video_name, 'sentence_en'].values[0] # MER2024

        character_line = "The person in video says: {}. ".format(sentence)
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))
        # print(instruction)
        
        return {
            "image": image,
            "video_features": video_features,
            "instruction_input": instruction,
            "answer": caption,
            "emotion": emotion,
            "image_id": video_name
        }
    
    def extract_frame(self, video_path):
        # Open the video file
        video_capture = cv2.VideoCapture(video_path)
        # Read the first frame
        success, frame = video_capture.read()
        if not success:
            raise ValueError("Failed to read video file:", video_path)
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Release the video capture object
        video_capture.release()
        return frame_rgb

    def get(self, video_name):
        # FaceMAE feature
        FaceMAE_feats_path = os.path.join(self.file_path, 'mae_340_23_UTT', video_name + '.npy') # MER2024
        FaceMAE_feats = torch.tensor(np.load(FaceMAE_feats_path))

        # VideoMAE feature
        VideoMAE_feats_path = os.path.join(self.file_path, 'maeVideo_399_23_UTT', video_name + '.npy')
        VideoMAE_feats = torch.tensor(np.load(VideoMAE_feats_path))

        # Audio feature
        Audio_feats_path = os.path.join(self.file_path, 'HL_23_UTT', video_name + '.npy')
        Audio_feats = torch.tensor(np.load(Audio_feats_path))

        return FaceMAE_feats, VideoMAE_feats, Audio_feats
