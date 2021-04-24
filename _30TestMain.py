#-*- coding:utf-8 _*-
"""
Testing
@Author  : Xiaoqi Cheng
@Time    : 2021/1/15 9:52
"""
import torch
from _99Normalization import *
from _31TestMultiPipeDatasetLoader import *
from _03FCN import *
from _21CalEvaluationMetrics import *

# %% Load dataset and model
FolderPath = 'METCD'
TestDataset, TestDataLoader = PipeDatasetLoader(FolderPath, Width = 2, ShowSample=False)
Model = Net(InputChannels=9, OutputChannels=1, InitFeatures=32, WithActivateLast=True, ActivateFunLast=torch.sigmoid).cuda()
Model.load_state_dict(torch.load('InputME_Width2_BCE-DC_0/0700.pt', map_location = 'cuda'))
SaveFolder = 'TestResult'

# %% Testing
Model.eval()
torch.set_grad_enabled(False)

for Iter, (Input, TMImg, SampleName) in enumerate(TestDataLoader):
	print(SampleName)
	Input = torch.cat(Input, dim=1)
	InputImg = Input.float().to('cuda')
	OutputImg = Model(InputImg)
	# Generate result image
	OutputImg = OutputImg.cpu().numpy()[0, 0]
	OutputImg = (OutputImg*255).astype(np.uint8)
	TMImg = TMImg.numpy()[0][0]
	TMImg = (Normalization(TMImg) * 255).astype(np.uint8)
	ResultImg = cv2.cvtColor(TMImg, cv2.COLOR_GRAY2RGB)
	# ResultImg[...,2] = OutputImg

	contour_points = np.argwhere(OutputImg > 100)
	ResultImg[contour_points[:, 0], contour_points[:, 1], 2] = 255

	# plt.imshow(OutputImg)
	# plt.show()
	os.makedirs(SaveFolder, exist_ok=True)
	cv2.imwrite(os.path.join(SaveFolder, SampleName[0] + '.png'), ResultImg)


