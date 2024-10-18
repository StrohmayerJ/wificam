import math
from torch.utils.data import Dataset
from skimage import io
import numpy as np
from tqdm import tqdm
import albumentations as A
import torch
from torchvision import transforms
import pandas as pd
import yaml
import random
import os

# select valid subcarriers
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_index += [i for i in range(6, 32)]
csi_vaid_subcarrier_index += [i for i in range(33, 59)]
CSI_SUBCARRIERS = len(csi_vaid_subcarrier_index)


def encode_time(x, L, window_size):
    window_size *= 3
    frequencies = np.array([2**i for i in range(L)])
    x = x/window_size
    pos_enc = np.concatenate([np.sin(frequencies[:, None] * np.pi * x),
                              np.cos(frequencies[:, None] * np.pi * x)], axis=0)
    return pos_enc


class WificamDataset(Dataset):
    def __init__(self, dataPath, augPath, windowSize, frequence_L,  random_sample=True, temporal_encoding=False):
        self.random_sample = random_sample
        self.dataPath = dataPath
        folderEndIndex = dataPath.rfind('/')
        self.dataFolder = self.dataPath[0:folderEndIndex]
        self.imagePath = os.path.dirname(dataPath)
        self.windowSize = windowSize
        assert windowSize % 2 == 1
        self.windowSizeH = math.ceil(windowSize/2)
        self.params = None
        self.augPath = augPath
        self.csiMean = 0
        self.csiStd = 0
        self.imgMean = [0, 0, 0]
        self.imgStd = [0, 0, 0]
        self.temporal_encoding = temporal_encoding
        self.L = frequence_L

        # get statistics
        if '320' in dataPath:
            statistics = np.genfromtxt(os.path.dirname((os.path.dirname(os.path.dirname(dataPath)))) + "/statistics320.csv", delimiter=',', dtype=np.float32)
        else:
            statistics = np.genfromtxt(os.path.dirname((os.path.dirname(os.path.dirname(dataPath)))) + "/statistics640.csv", delimiter=',', dtype=np.float32)
        self.csiMean = statistics[0][0]
        self.csiStd = statistics[1][0]
        self.imgMean = statistics[2]
        self.imgStd = statistics[3]

        # read augmentation parameters from yaml file
        if self.augPath != '':
            with open(self.augPath, 'r') as file:
                self.params = yaml.safe_load(file)

        # read CSI data from .csv file
        data = pd.read_csv(dataPath)
        csi = data['data']

        # create or load complex CSI cache
        if os.path.exists(self.dataFolder + f"/csiComplex.npy"):
            csiComplex = np.load(self.dataFolder + f"/csiComplex.npy")
        else:
            csiComplex = np.zeros(
                [len(csi), CSI_SUBCARRIERS], dtype=np.complex64)
            for s in tqdm(range(len(csi))):
                for i in range(CSI_SUBCARRIERS):
                    sample = csi[s][1:-1].split(',')
                    sample = np.array([int(x) for x in sample])
                    csiComplex[s][i] = complex(
                        sample[csi_vaid_subcarrier_index[i] * 2], sample[csi_vaid_subcarrier_index[i] * 2 - 1])
            np.save(self.dataFolder + f"/csiComplex.npy", csiComplex)

        # get image ids
        self.ids = data['id']

        # extract amplitudes from complex CSI
        self.csiAmplitudes = np.abs(csiComplex)

        # compute number of samples excluding border regions
        self.dataSize = len(self.csiAmplitudes)-self.windowSize

    # pixel-wise dropout augmentation
    def pixelwiseDropout(self, img):
        mask = np.random.choice([0, 1], size=img.shape, p=[
                                self.params['pixelWiseDropout'], 1 - self.params['pixelWiseDropout']])
        img_copy = img.copy()
        if np.random.rand() < self.params['pixelWiseDropoutZeroPixels']:
            img_copy *= mask  # replace with 0
        else:
            # replace with channel mean
            img_copy = img_copy * mask + np.mean(img_copy) * (1 - mask)
        return img_copy

    # column-wise dropout augmentation
    def columnwiseDropout(self, img):
        mask = np.random.choice([0, 1], size=img.shape[1:], p=[
                                self.params['columnWiseDropout'], 1 - self.params['columnWiseDropout']])
        img_copy = img.copy()
        if np.random.rand() < self.params['columnWiseDropoutZeroPixels']:
            img_copy *= mask[None, :]  # replace with 0
        else:
            # replace with channel mean
            img_copy = img_copy * mask[None, :] + \
                np.mean(img_copy) * (1 - mask[None, :])
        return img_copy

    def __len__(self):
        return self.dataSize

    def __getitem__(self, index):
        index = index + self.windowSizeH # add index offset to avoid border regions
        spectrogram = self.csiAmplitudes[index-self.windowSizeH:index+self.windowSizeH-1] # get amplitude spectrogram
        spectrogram = np.transpose(spectrogram, (1, 0))

        # augmentations
        if self.augPath != '':
            if np.random.rand() < self.params['amplitudeProbability']:
                spectrogram = A.RandomBrightnessContrast(p=1, brightness_limit=(
                    -self.params['amplitudeRange'], self.params['amplitudeRange']), contrast_limit=0, always_apply=True)(image=spectrogram)['image']
            if np.random.rand() < self.params['pixelWiseDropoutProbability']:
                spectrogram = self.pixelwiseDropout(spectrogram)
                spectrogram = spectrogram.astype(np.float32)
            if np.random.rand() < self.params['columnWiseDropoutProbability']:
                spectrogram = self.columnwiseDropout(spectrogram)
                spectrogram = spectrogram.astype(np.float32)

        # convert to torch tensor
        spectrogram_transforms = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(self.csiMean, self.csiStd)
        ])
        spectrogram = spectrogram_transforms(spectrogram)

        # get all image ids within spectrogram window
        imageIds = self.ids[index-self.windowSizeH:index+self.windowSizeH-1]
        # remove non-existing ids
        imageIds = [
            id for id in imageIds
            if os.path.isfile(self.imagePath + "/" + str(id) + '.png')
        ]

        # pick central image id or perform random smapling
        if self.random_sample:
            id = random.choice(imageIds)
        else:
            id = imageIds[len(imageIds)//2]
        image_index = imageIds.index(id)

        # load image
        image = io.imread(self.imagePath + "/" + str(id) + '.png')
        # resize
        image = A.Resize(128, 128, always_apply=True)(image=image)['image']  # <- set whatever image size you want to use
        # convert to torch tensor
        image = torch.tensor(image, dtype=torch.float)/255.0
        # channel first
        image = image.permute(2, 0, 1)
        # normalize with channel means and stds
        image = transforms.Normalize(self.imgMean, self.imgStd)(image)

        # temporal encoding
        spectrogram_tenc = np.array([index if not self.temporal_encoding else 0])
        spectrogram_tenc = torch.tensor(encode_time(spectrogram_tenc,self.L,self.windowSize)).unsqueeze(0).float()
        image_tenc = np.array([image_index if not self.temporal_encoding else 0])
        image_tenc = torch.tensor(encode_time(image_tenc,self.L,self.windowSize)).unsqueeze(0).float()
        tenc = torch.cat((spectrogram_tenc, image_tenc), dim=2)
        return (tenc, spectrogram), (tenc, image)

