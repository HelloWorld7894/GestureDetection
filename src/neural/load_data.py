from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader

import os
import mediapipe_pretrained
import random

class RockPaperScissorsDataset(Dataset):
    def __init__(self, train_to_valid_ratio=5):
        self.dataset_paths = ["/media/work/Workspace/Projects/GestureDetection/src/neural/dataset/paper",
        "/media/work/Workspace/Projects/GestureDetection/src/neural/dataset/rock",
        "/media/work/Workspace/Projects/GestureDetection/src/neural/dataset/scissors"]

        self.y, self.X = self.read_data(train_to_valid_ratio)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def read_data(self, ratio):
        path_all = [[], [], []]

        train_paths_all = []
        valid_paths_all = []

        labels_train = [] #y
        coords_train = [] #X

        labels_valid = []
        coords_valid = []

        for idx, dataset_path in enumerate(self.dataset_paths):
            for filename in os.listdir(dataset_path):
                path = dataset_path + "/" + filename
                path_all[idx].append(path)

        for label in path_all:
            len_all = len(path_all[0]) + len(path_all[1]) + len(path_all[2])
            len_train = round(len_all / (ratio + 1))

            #get train dataset
            train_paths = random.sample(label, len_train)
            valid_paths = list(set(label) - set(train_paths))

            train_paths_all.extend(train_paths)
            valid_paths_all.extend(valid_paths)
        
            #TODO: later => X = mediapipe_pretrained.detect_hand_pretrained(path)
                
        for train_path in train_paths_all:
            X = mediapipe_pretrained.detect_hand_pretrained(train_path)

            coords_train.append(X)
            coords_valid
        
        for valid_path in valid_paths_all:
            X = mediapipe_pretrained.detect_hand_pretrained(train_path)



train_data, valid_data = RockPaperScissorsDataset()

train = DataLoader(train_data, batch_size=32, shuffle=True)
valid = DataLoader(valid_data, batch_size=32, shuffle=True)