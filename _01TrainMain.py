# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2020/10/24 9:33
"""
import logging, os, torch
from _99Timer import *
from _02MultiPipeDatasetLoader import *
from _03FCN import *
from _04LossFunction import *

def Train(SplitEpoch, SaveFolder, Width):
	# %% InitParameters
	BatchSize = 3
	Epochs = 700
	Lr = 0.01
	LrDecay = 0.9
	LrDecayPerEpoch = 70

	ValidPerEpoch = 5
	SaveEpoch = [Epochs]        # epochs need to save temporarily
	torch.cuda.set_device(0)
	Device = torch.device('cuda:0')
	BCELossWeightCoefficient = 2

	# %% Load Multi-exposure tube contour dataset (METCD)
	print('\n\n\n**************SaveFolder*****************\n')
	os.makedirs(SaveFolder, exist_ok=SaveFolder)
	logging.basicConfig(filename=os.path.join(SaveFolder, 'log.txt'), filemode='w', level=logging.WARNING, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d-%H:%M:%S')

	FolderPath = 'METCD'
	TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(FolderPath, TrainBatchSize=BatchSize, ValBatchSize=BatchSize,
	                                                                             TrainNumWorkers=2, ValNumWorkers=1, Width=Width)
	Model = Net(InputChannels=9, OutputChannels=1, InitFeatures=32, WithActivateLast=True, ActivateFunLast=torch.sigmoid).to(Device)

	# %% Init optimizer and learning rate
	CriterionBCELoss, CriterionDilateContourLoss = LossAdam(Device)
	for Epoch in range(1, Epochs + 1):
		End = timer(8)
		if Epoch == 1:
			Optimizer = torch.optim.Adam(Model.parameters(), lr=Lr)
			LrScheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=LrDecayPerEpoch, gamma=LrDecay)

		# %% Training
		Model.train()
		# torch.cuda.empty_cache()
		TrainLoss = 0
		BCELoss = 0
		DCLoss = 0
		print('Epoch:%d, LR:%.8f ' % (Epoch, LrScheduler.get_lr()[0]), end='>> ', flush=True)
		for Iter, (InputImgs, Label, TMImg, SampleName) in enumerate(TrainDataLoader):
			print(Iter, end=' ', flush=True)
			InputImg = torch.cat(InputImgs, dim=1)
			InputImg = InputImg.float().to(Device)
			Label = Label.float().to(Device)
			Weight = Label * (BCELossWeightCoefficient - 1) + 1
			CriterionBCELoss.weight = Weight
			Optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				OutputImg = Model(InputImg)
				BatchBCELoss = CriterionBCELoss(OutputImg, Label)
				BatchDCLoss = CriterionDilateContourLoss(OutputImg, Label)
				if Epoch > SplitEpoch:
					BatchLoss = BatchBCELoss + BatchDCLoss
				else:
					BatchLoss = BatchBCELoss
				BatchLoss.backward()
				Optimizer.step()
				TrainLoss += BatchLoss.item()
				BCELoss += BatchBCELoss.item()
				DCLoss += BatchDCLoss.item()
		AveTrainLoss = (TrainLoss * BatchSize) / TrainDataset.__len__()
		AveBCELoss = (BCELoss * BatchSize) / TrainDataset.__len__()
		AveDCLoss = (DCLoss * BatchSize) / TrainDataset.__len__()
		print(", Total loss is: %.6f" % float(AveTrainLoss))
		logging.warning('\tTrain\tEpoch:{0:04d}\tLearningRate:{1:08f}\tLoss:{2:08f}'.format(Epoch, LrScheduler.get_lr()[0], AveTrainLoss))
		print('\tAveBCELoss:{0:04f}\tAveDCLoss:{1:04f}'.format(float(AveBCELoss), float(AveDCLoss)))
		logging.warning('\tAveBCELoss:{0:04f}\tAveDCLoss:{1:04f}'.format(float(AveBCELoss), float(AveDCLoss)))
		End(SaveFolder+' Epoch')

		# %% Validation
		if Epoch % ValidPerEpoch == 0 or Epoch == 1:
			Model.eval()
			torch.cuda.empty_cache()
			ValLoss = 0
			BCELoss = 0
			DCLoss = 0
			print('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tValidate:', end='>>', flush=True)
			for Iter, (InputImgs, Label, TMImg, SampleName) in enumerate(ValDataLoader):
				print(Iter, end=' ', flush=True)
				InputImg = torch.cat(InputImgs, dim=1)
				InputImg = InputImg.float().to(Device)
				Label = Label.float().to(Device)
				Weight = Label * (BCELossWeightCoefficient - 1) + 1
				CriterionBCELoss.weight = Weight
				with torch.set_grad_enabled(False):
					OutputImg = Model(InputImg)
					BatchBCELoss = CriterionBCELoss(OutputImg, Label)
					BatchDCLoss = CriterionDilateContourLoss(OutputImg, Label)
					if Epoch > SplitEpoch:
						BatchLoss = BatchBCELoss + BatchDCLoss
					else:
						BatchLoss = BatchBCELoss
					ValLoss += BatchLoss.item()
					BCELoss += BatchBCELoss.item()
					DCLoss += BatchDCLoss.item()
			AveValLoss = (ValLoss * BatchSize) / ValDataset.__len__()
			AveBCELoss = (BCELoss * BatchSize) / ValDataset.__len__()
			AveDCLoss = (DCLoss * BatchSize) / ValDataset.__len__()
			print("Total loss is: %.6f" % float(AveValLoss))
			logging.warning('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tValid\tEpoch:{0:04d}\tLearningRate:{1:08f}\tLoss:{2:08f}'.format(Epoch, LrScheduler.get_lr()[0], AveValLoss))
			print('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tAveBCELoss:{0:04f}\tAveDCLoss:{1:04f}'.format(AveBCELoss, AveDCLoss))
			logging.warning('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tAveBCELoss:{0:04f}\tAveDCLoss:{1:04f}'.format(AveBCELoss, AveDCLoss))

		# %% Saving
		if Epoch in SaveEpoch:
			torch.save(Model.state_dict(), os.path.join(SaveFolder, '{0:04d}.pt'.format(Epoch)))
			print("Save path:", os.path.join(SaveFolder, '{0:04d}.pt'.format(Epoch)))
		LrScheduler.step()

	log = logging.getLogger()
	for hdlr in log.handlers[:]:
		if isinstance(hdlr, logging.FileHandler):
			log.removeHandler(hdlr)

if __name__ == '__main__':
	torch.backends.cudnn.benchmark = True
	Widths = [2]        # seting contour width of labels
	SplitEpoch = 420    # Before SplitEpoch, BCE Loss used; After SplitEpoch, BCE-DC Loss used
	for Width in Widths:
		for Count in range(0,10):
			SaveFolder = 'InputME_Width{0:d}_BCE-DC_{1:d}'.format(Width, Count)
			Train(SplitEpoch, SaveFolder, Width)


