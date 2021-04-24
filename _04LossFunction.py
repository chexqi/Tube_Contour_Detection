# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2020/10/24 20:50
"""
import numpy as np
import torch.nn as nn
import torch, cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

class DilateContourLoss(nn.Module):
	def __init__(self):
		super(DilateContourLoss, self).__init__()
		self.kernel = np.ones((3, 3), np.uint8)

	def forward(self, y_pred, y_true):  # BS*1*256*256
		assert y_pred.size() == y_true.size()
		Dilate_y_pred = F.max_pool2d(y_pred, kernel_size=3, stride=1, padding=1)#*2
		MissImg = torch.clamp(y_true - Dilate_y_pred, 0, 1)
		Dilate_y_true = F.max_pool2d(y_true, kernel_size=3, stride=1, padding=1)
		RedunImg = torch.clamp(y_pred - Dilate_y_true, 0, 1)
		Loss = (MissImg.sum() + RedunImg.sum()) / y_true.sum()
		return Loss

def LossAdam(Device):
	CriterionBCELoss = nn.BCELoss().to(Device)
	CriterionDilateContourLoss = DilateContourLoss().to(Device)
	return CriterionBCELoss, CriterionDilateContourLoss
