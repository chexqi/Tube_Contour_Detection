#-*- coding:utf-8 _*-
"""
Validatation
@Author  : Xiaoqi Cheng
@Time    : 2021/1/11 9:52
"""
import torch,glob
from _99Normalization import *
from _99SaveLoad import *
from _02MultiPipeDatasetLoader import *
from _03FCN import *
from _21CalEvaluationMetrics import *

SaveFolder = "InputME_Width2_BCE-DC_0"
Width = 2

# %% Load METCD and model
FolderPath = 'METCD'
TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(FolderPath, TrainBatchSize=1, ValBatchSize=1,
                                                                             TrainNumWorkers=0, ValNumWorkers=0, Width=Width)
ModelNames = ['0700']

for ModelName in ModelNames:
	SaveFilePath = os.path.join(SaveFolder, 'result' + ModelName + '.txt')
	if os.path.exists(SaveFilePath):
		print(SaveFilePath+' already exist!')
		continue

	Model = Net(InputChannels=9, OutputChannels=1, InitFeatures=32, WithActivateLast=True, ActivateFunLast=torch.sigmoid).cuda()
	Model.load_state_dict(torch.load(os.path.join(SaveFolder, ModelName+'.pt'), map_location = 'cuda'))

	# %% Evaluation
	Model.eval()
	torch.set_grad_enabled(False)
	OutputS = []
	LabelS = []
	for Iter, (Input, Label, TMImg, SampleName) in enumerate(ValDataLoader):
		# print(SampleName)
		Input = torch.cat(Input, dim=1)
		InputImg = Input.float().to('cuda')
		OutputImg = Model(InputImg)
		# Record
		Output = OutputImg.detach().cpu().numpy()[0]
		Label = Label.detach().cpu().numpy()[0]
		OutputS.append(Output)
		LabelS.append(Label)

		OutputImg = OutputImg.cpu().numpy()[0, 0]
		OutputImg = (OutputImg*255).astype(np.uint8)
		TMImg = TMImg.numpy()[0][0]
		TMImg = (Normalization(TMImg) * 255).astype(np.uint8)
		ResultImg = cv2.cvtColor(TMImg, cv2.COLOR_GRAY2RGB)
		LabelImg = (Normalization(Label[0]) * 255).astype(np.uint8)
		ResultImg[..., 2] = cv2.add(ResultImg[..., 2], OutputImg)

	# %% Calculate evaluation metrics
	OutputFlatten = np.vstack(OutputS).ravel()
	LabelFlatten = np.vstack(LabelS).ravel()

	_, _, MF, mAP = PRC_mAP_MF(LabelFlatten, OutputFlatten, ShowPRC = False)
	print('MF:', MF)
	print('mAP:', mAP)

	# DIA-ODS
	DIA_ODS, OptThreshold = DIA_ODS(OutputS, LabelS, ShowCurve = False)
	print('DIA-ODS:',DIA_ODS, ' at threshold:', OptThreshold)

	with open(SaveFilePath, 'w') as f:
		# f.write('AUC: '+str(AUC)+'\n')
		f.write('MF: '+str(MF)+'\n')
		f.write('mAP: '+str(mAP)+'\n')
		f.write('DIA-ODS: '+str(DIA_ODS)+' at threshold: '+str(OptThreshold)+'\n')


