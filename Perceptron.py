
import numpy as np
import pandas as pd


class Perceptron(object):

	def __init__(self):
		self.learning_step = 1
		self.max_iter = 10

	def train(self, data_x, data_y):
		print("start training with training set...")
		b = np.asarray([1] * (data_x.shape[0]))
		data_x = np.vstack((data_x.transpose(), b))
		self.w = [0] * data_x.shape[1]
		print(self.w)
		print(data_x)

	
		time = 0
		pool = [i for i in range(0,data_x.shape[0])]
		while time < self.max_iter:
			i = np.random.choice(pool)

			loss = data_x[i].dot(self.w) * data_y[i]
			print("===============================")
			print("current loss: " + str(loss))
			print("x: " + str(data_x[i]) + " y: " + str(data_y[i]))
			print("current linear model: " + str(self.w))
	
			if loss <= 0:
				self.w += self.learning_step * data_y[i] * data_x[i]
				pool = [i for i in range(0,data_x.shape[0])]
			else:
				pool.remove(i)
				if pool == []:
					break

			time += 1
			if time >= self.max_iter:
				break 
		print("Total iteration = " + str(time))







if __name__ == "__main__":
    
    # data prepocessing 
	train_set = np.asarray([[3,3,1],[4,3,1],[1,1,-1]])

	print(train_set)
	train_x = train_set[:, :-1]
	train_y = train_set[:, -1]


	# model training
	model = Perceptron()
	model.train(train_x, train_y)
	print(model.w)
