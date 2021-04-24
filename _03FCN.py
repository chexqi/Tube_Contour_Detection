# -*- coding:utf-8 _*-
"""
Our FCN is changed based on U-Net
O. Ronneberger, P. Fischer, T. Brox, U-net: convolutional networks for biomedical image segmentation, in: Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention, 2015, pp. 234-241.
@Author  : Xiaoqi Cheng
@Time    : 2020/10/24 16:00
"""
from collections import OrderedDict
import torch, torchsummary
import torch.nn as nn
import numpy as np
import _99Timer

class Net(nn.Module):
	def __init__(self, InputChannels=1, OutputChannels=1, InitFeatures=64, WithActivateLast=True, ActivateFunLast=None):
		super(Net, self).__init__()
		IF = InitFeatures
		BigKernel = 4
		DoubleFeatures = 1      # Feature chanels increasing speed
		self.WithActivateLast = WithActivateLast  # True: Add activation function，False: Without activation function
		self.ActivateFunLast = ActivateFunLast
		self.EB1 = self.Block(InputChannels, IF, Name="EB1")
		self.MP1 = nn.MaxPool2d(kernel_size=BigKernel, stride=BigKernel)
		self.EB2 = self.Block(IF, IF * 2 * DoubleFeatures, Name="EB2")
		self.MP2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.EB3 = self.Block(IF * 2 * DoubleFeatures, IF * 4 * DoubleFeatures, Name="EB3")
		self.MP3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.EB4 = self.Block(IF * 4 * DoubleFeatures, IF * 8 * DoubleFeatures, Name="EB4")
		self.MP4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.EB5orDB5 = self.Block(IF * 8 * DoubleFeatures, IF * 16 * DoubleFeatures, Name="EB5orDB5")

		self.CT4 = nn.ConvTranspose2d(IF * 16 * DoubleFeatures, IF * 8 * DoubleFeatures, kernel_size=2, stride=2)
		self.DB4 = self.Block((IF * 8 * DoubleFeatures) * 2, IF * 8 * DoubleFeatures, Name="DB4")
		self.CT3 = nn.ConvTranspose2d(IF * 8 * DoubleFeatures, IF * 4 * DoubleFeatures, kernel_size=2, stride=2)
		self.DB3 = self.Block((IF * 4 * DoubleFeatures) * 2, IF * 4 * DoubleFeatures, Name="DB3")
		self.CT2 = nn.ConvTranspose2d(IF * 4 * DoubleFeatures, IF * 2 * DoubleFeatures, kernel_size=2, stride=2)
		self.DB2 = self.Block((IF * 2 * DoubleFeatures) * 2, IF * 2 * DoubleFeatures, Name="DB2")
		self.CT1 = nn.ConvTranspose2d(IF * 2 * DoubleFeatures, IF, kernel_size=BigKernel, stride=BigKernel)
		self.DB1 = self.Block(IF * 2, IF, Name="DB1")
		self.Mapping = nn.Conv2d(in_channels=IF, out_channels=OutputChannels, kernel_size=1)

	def forward(self, x):
		EB1 = self.EB1(x)
		EB2 = self.EB2(self.MP1(EB1))
		EB3 = self.EB3(self.MP2(EB2))
		EB4 = self.EB4(self.MP3(EB3))
		EB5orDB5 = self.EB5orDB5(self.MP4(EB4))
		DB4 = self.CT4(EB5orDB5)
		DB4 = torch.cat((DB4, EB4), dim=1)
		DB4 = self.DB4(DB4)
		DB3 = self.CT3(DB4)
		DB3 = torch.cat((DB3, EB3), dim=1)
		DB3 = self.DB3(DB3)
		DB2 = self.CT2(DB3)
		DB2 = torch.cat((DB2, EB2), dim=1)
		DB2 = self.DB2(DB2)
		DB1 = self.CT1(DB2)
		DB1 = torch.cat((DB1, EB1), dim=1)
		DB1 = self.DB1(DB1)
		if self.WithActivateLast:
			return self.ActivateFunLast(self.Mapping(DB1))
		else:
			return self.Mapping(DB1)

	# %% Ecode Block or Decode Block in different spatial scales
	def Block(self, InputChannels, OutputChannels, Name):
		return nn.Sequential(OrderedDict([
			(Name + "_Conv1", nn.Conv2d(in_channels=InputChannels, out_channels=OutputChannels, kernel_size=3, padding=1, bias=False)),
			(Name + "_Norm1", nn.BatchNorm2d(num_features=OutputChannels)),
			(Name + "_Relu1", nn.ReLU(inplace=True)),
			# (Name + "_Drop1", nn.Dropout2d(0.5)),
			(Name + "_Conv2", nn.Conv2d(in_channels=OutputChannels, out_channels=OutputChannels, kernel_size=3, padding=1, bias=False)),
			(Name + "_Norm2", nn.BatchNorm2d(num_features=OutputChannels)),
			(Name + "_Relu2", nn.ReLU(inplace=True)),
			# (Name + "_Drop2", nn.Dropout2d(0.01)),
		]))


if __name__ == '__main__':
	Model = Net(InputChannels=1, OutputChannels=1, InitFeatures=32, WithActivateLast=True, ActivateFunLast=torch.sigmoid).cuda()
	torchsummary.summary(Model, (1, 1024, 1280), 1, 'cuda')
	exit()

	Optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)
	Input = torch.randn((3, 6, 1024, 1280)).cuda()
	Target = torch.empty((3, 1, 1024, 1280), dtype=torch.long).random_(2).cuda()

	LossFun = nn.BCELoss()
	Output = Model(Input)

	print(Output.shape)
	print(Target.shape)

	BatchLoss = LossFun(Output.float(), Target.float())
	print(BatchLoss)

	BatchLoss.backward()
	Optimizer.step()
