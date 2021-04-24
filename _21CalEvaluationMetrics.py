#-*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2021/1/12 21:08
"""
import sys, cv2
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import numpy.random as r
from tqdm import tqdm
import sklearn.metrics as m
import matplotlib.pyplot as plt

def PRC_mAP_MF(LabelFlatten, OutputFlatten, ShowPRC = False):
	Precision, Recall, th = m.precision_recall_curve(LabelFlatten, OutputFlatten)
	F1ScoreS = 2 * (Precision * Recall) / ((Precision + Recall) + sys.float_info.min)
	MF = F1ScoreS[np.argmax(F1ScoreS)]  # Maximum F-measure at optimal dataset scale
	mAP = m.average_precision_score(LabelFlatten, OutputFlatten)
	if ShowPRC:
		plt.figure('Precision Recall curve')
		plt.plot(Recall, Precision)
		plt.ylim([0.0, 1.0])
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		# plt.show()
	return Recall, Precision, MF, mAP

# Dilate inaccuracy at optimal dataset scale(DIA_ODS)
def DIA_ODS(OutputS, LabelS, ShowCurve = False):
	DIA_ODSs = []
	kernel = np.ones((3, 3), np.uint8)
	for threshold in range(0, 255, 1):
		MissSum = 0
		RedunSum = 0
		TrueSum = 0
		for (Output, Label) in zip(OutputS, LabelS):
			PredImg = (Output[0] * 255).astype(np.uint8)
			_, PredImg = cv2.threshold(PredImg, threshold, 255, cv2.THRESH_BINARY)
			TrueImg = (Label[0] * 255).astype(np.uint8)
			# Missing contours
			Dilatey_pred_Img = cv2.dilate(PredImg, kernel)
			MissImg = cv2.subtract(TrueImg, Dilatey_pred_Img)
			# Redundant  contour
			Dilatey_true_Img = cv2.dilate(TrueImg, kernel)
			RedunImg = cv2.subtract(PredImg, Dilatey_true_Img)

			# fig, axs = plt.subplots(2,2)
			# axs = axs.flatten()
			# axs[0].imshow(MissImg)
			# axs[0].set_title('MissImg')
			# axs[1].imshow(RedunImg)
			# axs[1].set_title('RedunImg')
			# axs[2].imshow(PredImg)
			# axs[2].set_title('PredImg')
			# axs[3].imshow(TrueImg)
			# axs[3].set_title('TrueImg')
			# plt.show()
			MissSum = MissSum + MissImg.sum()
			RedunSum = RedunSum + RedunImg.sum()
			TrueSum = TrueSum + TrueImg.sum()
		DIA = (MissSum + RedunSum) / TrueSum
		DIA_ODSs.append(DIA)
	if ShowCurve:
		plt.plot(DIA_ODSs)
		plt.show()
	DIA_ODS = np.min(DIA_ODSs)
	OptThreshold = np.argmin(DIA_ODSs)
	return DIA_ODS, OptThreshold