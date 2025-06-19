from moviepy.editor import VideoFileClip
import os
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor
import numpy as np
import tqdm

def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    # audio.write_audiofile("audio.wav")

    audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
    audio.write_audiofile(audio_path, fps=16000, codec='pcm_s16le', ffmpeg_params=['-ac', '1'])
    samples, sr = sf.read(audio_path)
    return samples, sr

def extract_hu_features():
    # Assuming hu_features is a function that extracts features from audio samples
    # Replace this with the actual implementation of hu_features
    videos_dir = "data/video_data"
    save_dir = "data/hu_features"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for video_file in tqdm.tqdm(os.listdir(videos_dir)):
        video_path = os.path.join(videos_dir, video_file)
        samples, sr = extract_audio_from_video(video_path)

        model_file = "checkpoints/transformer/hubert-large-ll60k"
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file)
        input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_values
        
        from transformers import HubertModel
        hubert_model = HubertModel.from_pretrained(model_file)
        hubert_model.eval()
        with torch.no_grad():
            hidden_states = hubert_model(input_values, output_hidden_states=True).hidden_states # tuple of (B, T, D)
            audio_feature = torch.stack(hidden_states)[[-1]].sum(dim=0)  # sum, (B, T, D)
            audio_feature = audio_feature[0].detach().unsqueeze(0)
            audio_feature = torch.mean(audio_feature, dim=1, keepdim=True)

        # save the feature as numpy array
        feature_path = os.path.join(save_dir, video_file.replace('.mp4', '.npy'))
        
        np.save(feature_path, audio_feature.cpu().numpy())
    
if __name__ == "__main__":
    extract_hu_features()