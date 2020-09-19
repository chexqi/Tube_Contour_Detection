#-*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2019/10/25 10:28
"""
import numpy as np

def Normalization(Array):
	'''
	:param Array:
	:return:
	'''
	min = np.min(Array)
	max = np.max(Array)
	if max-min == 0:
		return Array
	else:
		return (Array - min) / (max - min)