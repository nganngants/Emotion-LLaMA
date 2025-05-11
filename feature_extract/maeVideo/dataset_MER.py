# @Time    : 6/26/23 6:51 PM
# @Author  : bbbdbbb
# @File    : dataset_MER.py
# @Description : dataset to read MER2023

import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
from .video_transform import *
import numpy as np
import cv2

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(self, list_file, face_dir, num_segments, duration, mode, transform, image_size):
        self.list_file = list_file
        self.face_dir = face_dir
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()
        pass

    def _parse_list(self):
        # check the frame number is large >=16:
        # form is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        # tmp = [item for item in tmp if int(item[1]) >= 16]
        tmp = [item for item in tmp]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # split all frames into seg parts, then select frame in each part randomly
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        # split all frames into seg parts, then select frame in the mid of each part
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        video_name = record.path.split('/')[-1]
        # print(video_name)

        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)

        images = self.get(record, segment_indices)
        images = images.view((16, 3) + images.size()[-2:]).transpose(0, 1)
        # print("images shape: ", images.shape)

        return  images, video_name
        # return  (pixel_values, mask)

    def get(self, record, indices):
        video_name = record.path.split('/')[-1]

        ## 绝对路径
        # # video_frames_path = glob.glob(os.path.join(record.path, '*.jpg'))
        # video_frames_path = glob.glob(os.path.join(record.path, '*.bmp'))
        # video_frames_path.sort()

        ## 相对路径
        video_path = self.face_dir
        video_frames_path = glob.glob(os.path.join(video_path, video_name, '*.bmp'))

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1

        images = self.transform(images)

        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        # return images, record.label
        return images

    def __len__(self):
        return len(self.video_list)


class MESCVideoDataset(data.Dataset):
    def __init__(self, face_dir, num_segments, duration, mode, transform, image_size):
        self.face_dir = face_dir
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()
        pass

    def _parse_list(self):
        self.video_files = glob.glob(os.path.join(self.face_dir, '*.mp4'))

        print(('video number:%d' % (len(self.video_files))))

    def _get_train_indices(self, num_frames):
        # split all frames into seg parts, then select frame in each part randomly
        average_duration = (num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(randint(num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets
    
    def _get_test_indices(self, num_frames):
        # split all frames into seg parts, then select frame in the mid of each part
        if num_frames > self.num_segments + self.duration - 1:
            tick = (num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets
    
    def __getitem__(self, index):
        file = self.video_files[index]
        video_name = file.split('/')[-1]
        # print(video_name)

        images = []

        if self.mode == 'train':
            return NotImplementedError("Train mode is not implemented for MESCVideoDataset")
        elif self.mode == 'test':
            # Open the video file using OpenCV, get the number of frames, get the indices, get the frames images and transform them
            cap = cv2.VideoCapture(file)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if num_frames == 0:
                print("Warning: number of frames of video {} should not be zero.".format(video_name))
                return None, video_name
            
            if num_frames < 16:
                # read all frames, duplicate the last frames
                for i in range(num_frames):
                    ret, frame = cap.read()
                    if ret:
                        images.append(frame)
                    else:
                        print(f"Warning: Could not read frame {i} from video {video_name}")
                images.append(images[-(16 - num_frames):])
            else:
                segment_indices = self._get_test_indices(num_frames)

            # Read the frames from the video file
            current_frame = 0
            for start_idx in segment_indices:
                # If we need to skip frames to reach the target index
                start_idx = int(start_idx)
                if start_idx > current_frame:
                    for _ in range(start_idx - current_frame):
                        cap.grab()
                    current_frame = start_idx
                for offset in range(self.duration):
                    if current_frame + offset >= num_frames:
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Add to our final list
                    frame = Image.fromarray(frame)
                    images.append(frame)
                
                current_frame = start_idx + self.duration

        # print("images shape: ", len(images), images[0].shape)
        images = self.transform(images)

        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        images = images.view((16, 3) + images.size()[-2:]).transpose(0, 1)

        return  images, video_name
        # return  (pixel_values, mask)
    
    def __len__(self):
        return len(self.video_files)
    

def train_data_loader(face_dir):
    image_size = 224
    # GroupRandomSizedCrop(image_size),
    train_transforms = torchvision.transforms.Compose([
                                                       GroupResize(image_size),
                                                       GroupRandomHorizontalFlip(),
                                                       Stack(),
                                                       ToTorchFormatTensor()])
    train_data = MESCVideoDataset(face_dir=face_dir,
                              num_segments=8,
                              duration=2,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size)
    return train_data


def test_data_loader(face_dir):
    image_size = 224
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    
    # "/home/amax/big_space/datasets/DFEW/dataset-process/my-process/DFEW_set_1_train.txt"
    # "/home/amax/big_space/datasets/MER2023/dataset-process/my-process/all_NCEV.txt"
    # "/home/amax/big_space/datasets/MER2024/dataset-process/my-process/MER2024_12065_NCE.txt"
    # "/home/amax/big_space/datasets/MER2024/EMER/EMER_332_NCE.txt"
    test_data = MESCVideoDataset(face_dir=face_dir,
                             num_segments=8,
                             duration=2,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size)
    return test_data

