# -*- coding:utf-8 _*-
"""
Load Multi-exposure tube contour dataset (METCD)
@Author  : Xiaoqi Cheng
@Time    : 2020/10/24 19:40
"""
import torch, os, cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

ImgResize = (1024,1280) # image size

# %% Data augmentation
TrainImgTransform = transforms.Compose([
	transforms.RandomAffine(degrees=(-45, 45), translate=(0.2, 0.2), scale=(1, 1.), shear=10),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	transforms.RandomResizedCrop(ImgResize, scale=(1., 1.), interpolation=Image.BILINEAR),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.46], std=[0.10]),
])
TrainLabelTransform = transforms.Compose([
	transforms.RandomAffine(degrees=(-45, 45), translate=(0.2, 0.2), scale=(1, 1.), shear=10),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	transforms.RandomResizedCrop(ImgResize, scale=(1., 1.), interpolation=Image.NEAREST),
	transforms.ToTensor(),
])

ValImgTransform = transforms.Compose([
	transforms.Resize(ImgResize),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.46], std=[0.10]),
])
ValLabelTransform = transforms.Compose([
	transforms.Resize(ImgResize, interpolation=Image.NEAREST),
	transforms.ToTensor(),
])

class PipeDataset(Dataset):
	def __init__(self, DatasetFolderPath, ImgTransform, LabelTransform, Width, ShowSample=False, ):
		self.DatasetFolderPath = DatasetFolderPath
		self.ImgTransform = ImgTransform
		self.LabelTransform = LabelTransform
		self.ShowSample = ShowSample
		self.SampleFolders = os.listdir(self.DatasetFolderPath)
		self.Width = Width

	def __len__(self):
		return len(self.SampleFolders)

	def __getitem__(self, item):
		ImgNames = ['002048.png', '004096.png', '008192.png', '016384.png',
		            '032768.png', '065536.png', '131072.png', '262144.png', '524288.png']
		# ImgNames = ['032768.png']
		# ImgNames = ['TonemapImg.png']
		SampleFolderPath = os.path.join(self.DatasetFolderPath, self.SampleFolders[item])
		MultiImgPaths = [os.path.join(SampleFolderPath, ImgName) for ImgName in ImgNames]
		MultiImgs = [Image.open(MultiImgPath) for MultiImgPath in MultiImgPaths]
		LabelImgPath = os.path.join(SampleFolderPath, 'EdgeLabel_%d.png' %self.Width)
		LabelImg = Image.open(LabelImgPath)
		TMImgPath = os.path.join(SampleFolderPath, 'TonemapImg.png')
		TMImg = Image.open(TMImgPath)

		# %% Ensure that the input data and the label have the same transformation
		seed = np.random.randint(2147483647)
		TranMultiImgs = []
		for MultiImg in MultiImgs:
			random.seed(seed)
			torch.manual_seed(seed)
			TranMultiImgs.append(self.ImgTransform(MultiImg))
		random.seed(seed)
		torch.manual_seed(seed)
		LabelImg = self.LabelTransform(LabelImg)
		random.seed(seed)
		torch.manual_seed(seed)
		TMImg = self.LabelTransform(TMImg)

		# %% Show Sample
		if self.ShowSample:
			plt.figure(self.SampleFolders[item])
			Img = TMImg.numpy()[0]
			Label = LabelImg.numpy()[0]
			Img = (Normalization(Img) * 255).astype(np.uint8)
			Label = (Normalization(Label) * 255).astype(np.uint8)
			Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
			Img[..., 2] = Label
			plt.imshow(Img)
			plt.show()
		return TranMultiImgs, LabelImg, TMImg, self.SampleFolders[item]

def PipeDatasetLoader(FolderPath, TrainBatchSize=3, ValBatchSize=1, TrainNumWorkers=0, ValNumWorkers=0, Width = 4, ShowSample=False):
	TrainFolderPath = os.path.join(FolderPath, 'Train')
	ValFolderPath = os.path.join(FolderPath, 'Val')
	TrainDataset = PipeDataset(TrainFolderPath, TrainImgTransform, TrainLabelTransform, Width, ShowSample, )
	ValDataset = PipeDataset(ValFolderPath, ValImgTransform, ValLabelTransform, Width, ShowSample, )
	TrainDataLoader = DataLoader(TrainDataset, batch_size=TrainBatchSize, shuffle=True, drop_last=False, num_workers=TrainNumWorkers, pin_memory=True)
	ValDataLoader = DataLoader(ValDataset, batch_size=ValBatchSize, shuffle=False, drop_last=False, num_workers=ValNumWorkers, pin_memory=True)
	return TrainDataset, TrainDataLoader, ValDataset, ValDataLoader

def Normalization(Array):
	min = np.min(Array)
	max = np.max(Array)
	if max - min == 0:
		return Array
	else:
		return (Array - min) / (max - min)


if __name__ == '__main__':
	FolderPath = 'METCD'
	TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(FolderPath, TrainBatchSize=1, ValBatchSize=1,
	                                                                             TrainNumWorkers=0, ValNumWorkers=0, Width = 2, ShowSample=True)
	for epoch in range(1):
		for i, (Imgs, Label, TMImg, SampleName) in enumerate(TrainDataLoader):
			Img = torch.cat(Imgs, dim=1)
			print(SampleName)
			print(Img.shape)
			print(Label.shape)


