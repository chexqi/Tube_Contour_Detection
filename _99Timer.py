#-*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2020/12/7 8:34
"""
import time

def timer(long = -1):
	start = time.time()
	def end(method_name="Unnamed function"):
		print(method_name + " took : " + str(time.time() - start)[0:long] + " seconds.")
		return

	return end

class Timer():
	def __init__(self,long = 6):
		self.long = long
		self.start = time.time()
	def __str__(self):      # 魔法方法 print实现时间打印
		return ("Time: " + str(time.time() - self.start)[0:self.long] + " seconds.")

if __name__ == '__main__':
	end = timer(long=8)
	time.sleep(1)
	end("Test")

	a = Timer(8)
	time.sleep(1)
	print(a)
	time.sleep(1)
	print(a)

