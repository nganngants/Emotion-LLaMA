from moviepy.editor import VideoFileClip
import os
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor
import numpy as np
import tqdm
from torch.nn.utils.rnn import pad_sequence
import json

def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    
    audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
    audio.write_audiofile(audio_path, fps=16000, codec='pcm_s16le', ffmpeg_params=['-ac', '1'])
    samples, sr = sf.read(audio_path)
    
    return samples, sr

def pad_audio_batch(audio_list, target_length=None):
    """Pad audio samples to the same length for batching"""
    if target_length is None:
        target_length = max(len(audio) for audio in audio_list)
    
    padded_audios = []
    for audio in audio_list:
        if len(audio) < target_length:
            # Pad with zeros
            padded_audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        else:
            # Truncate if longer
            padded_audio = audio[:target_length]
        padded_audios.append(padded_audio)
    
    return np.array(padded_audios)

def extract_hu_features_batch(batch_size=4, max_audio_length=160000):  # ~10 seconds at 16kHz
    videos_dir = "data/video_data"
    save_dir = "hu_features"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_file = "checkpoints/transformer/hubert-large-ll60k"
    from transformers import HubertModel
    hubert_model = HubertModel.from_pretrained(model_file)
    hubert_model.to('cuda')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file)
    hubert_model.eval()

    video_files = []
    jsonl_file = "data/train.jsonl"
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            video_files.append(data['path_to_vid_user_most_recent'][-1])
    
    # Process videos in batches
    for i in tqdm.tqdm(range(0, len(video_files), batch_size), desc="Processing batches"):
        torch.cuda.empty_cache()  # Clear GPU memory before processing a new batch
        batch_files = video_files[i:i + batch_size]
        batch_audio_data = []
        batch_video_names = []
        
        # Extract audio from all videos in the batch
        for video_file in batch_files:
            video_path = os.path.join(videos_dir, video_file)
            try:
                samples, sr = extract_audio_from_video(video_path)
                # Truncate very long audio to prevent memory issues
                if len(samples) > max_audio_length:
                    samples = samples[:max_audio_length]
                
                batch_audio_data.append(samples)
                batch_video_names.append(video_file)
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue
        
        if not batch_audio_data:
            continue
            
        # Pad audio samples to the same length
        padded_audio_batch = pad_audio_batch(batch_audio_data)
        
        # Process the batch through the feature extractor
        input_values = feature_extractor(
            padded_audio_batch, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        ).input_values.to('cuda')
        
        # Extract features for the entire batch
        with torch.no_grad():
            hidden_states = hubert_model(input_values, output_hidden_states=True).hidden_states
            # Get the last hidden state and sum across layers
            audio_features = torch.stack(hidden_states)[[-1]].sum(dim=0)  # (batch_size, T, D)
            
            # Average pool across time dimension
            audio_features = torch.mean(audio_features, dim=1, keepdim=True)  # (batch_size, 1, D)
        
        # Save features for each video in the batch
        for j, video_file in enumerate(batch_video_names):
            feature_path = os.path.join(save_dir, video_file.replace('.mp4', '.npy'))
            single_feature = audio_features[j].detach().cpu().numpy()
            np.save(feature_path, single_feature)

if __name__ == "__main__":
    # Choose one of the following:
    
    # Write audio first
    # video_files = []
    # jsonl_file = "data/test.jsonl"
    # with open(jsonl_file, 'r') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         video_files.append(data['path_to_vid_user_most_recent'][-1])
    # for video_file in tqdm.tqdm(video_files):
    #     video_path = os.path.join("data/video_data", video_file)
    #     try:
    #         extract_audio_from_video(video_path)
    #     except Exception as e:
    #         print(f"Error extracting audio from {video_file}: {e}")

    # Option 1: Fixed batch size
    extract_hu_features_batch(batch_size=64)
    
    # Option 2: Adaptive batch size (recommended)
    # extract_hu_features_adaptive_batch(initial_batch_size=8)