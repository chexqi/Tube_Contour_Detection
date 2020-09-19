#-*- coding:utf-8 _*-
"""
@author:Chexqi
@time: 2019/04/12
"""
import pickle

def Save(FileName, Data):
	with open(FileName, 'wb') as file:
		pickle.dump(Data, file)

def Load(FileName):
	with open(FileName, 'rb') as file:
		Data = pickle.load(file)
	return Data



