# *_*coding:utf-8 *_*
import os
import glob
from PIL import Image
import cv2
from skimage import io
import torch.utils.data as data


class FaceDataset(data.Dataset):
    def __init__(self, vid, face_dir, transform=None):
        super(FaceDataset, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        # Load the video from self.path and extract the first 16 frames as PIL Images
        video_capture = cv2.VideoCapture(self.path)
        
        frames = []
        for _ in range(16):
            try:
                success, frame = video_capture.read()
                if not success:
                    break
                # Convert BGR (OpenCV) to RGB and then to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            except Exception as e:
                print(f"Error reading frame: {e}")
                break

        video_capture.release()
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        img = self.frames[index]
        if self.transform is not None:
            img = self.transform(img)
        name = os.path.basename(self.path)[:-4] + f"_{index}"
        return img, name


class FaceDatasetForEmoNet(data.Dataset):
    def __init__(self, vid, face_dir, transform=None, augmentor=None):
        super(FaceDatasetForEmoNet, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.augmentor = augmentor
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        # Load the video from self.path and extract the first 16 frames
        video_capture = cv2.VideoCapture(self.path)
        
        frames = []
        for _ in range(16):
            try:
                success, frame = video_capture.read()
            except Exception as e:
                print(f"Error reading frame: {e}")
                break
            frames.append(frame)

        video_capture.release()
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        img = self.frames[index]
        if self.augmentor is not None:
            img = self.augmentor(img)[0]
        if self.transform is not None:
            img = self.transform(img)
        name = os.path.basename(self.path)[:-4] + f"_{index}"
        return img, name