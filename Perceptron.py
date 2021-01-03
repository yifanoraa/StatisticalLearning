
import numpy as np
import pandas as pd
import random


class Perceptron(object):

	def __init__(self):
		self.learning_step = 0.5
		self.max_iter = 50 
		self.w = []

	def train(self, data_x, data_y):
		print("start training with training set...")
		b = np.asarray([1] * (data_x.shape[0]))
		data_x = np.vstack((b, data_x.transpose())).transpose()
		self.w = [0] * data_x.shape[1]
	
		track = 0
		pocket_vector = [0] * data_x.shape[1]
		pocket_loss = 1000000
		print(data_x)

		while track < self.max_iter:
			index = random.choice(range(0, data_x.shape[0]))
			if self.check_sign(data_x[index], data_y[index], self.w) == -1:
				self.back_propagation(data_x[index], data_y[index])
				track += 1
			
			
			loss = self.loss(data_x, data_y, self.w)

			if loss < pocket_loss:
				pocket_vector = list(self.w)
				pocket_loss = loss


		# print("Total iteration = " + str(track))
		print("pocket_vector_loss = " + str(self.loss(data_x,data_y,pocket_vector)))
		print("pocket_loss = " + str(pocket_loss))
		self.w = pocket_vector
		return track


	def check_sign(self, x, y, weights):
		result = x.dot(weights)*y
		if result > 0:
			return 0
		else:
			return -1

	def back_propagation(self, x, y):
		print("--- back propagate ---")
		self.w += self.learning_step * y * x

	def loss(self, data_x, data_y, model):
		loss = 0
		for i in range(0, data_x.shape[0]):
			loss -= self.check_sign(data_x[i], data_y[i], model)

		return loss/data_x.shape[0]






if __name__ == "__main__":
    

	train_data = np.loadtxt('hw1_18_train.dat')
	test_data = np.loadtxt('hw1_18_test.dat')
	model = Perceptron()


	s = 0
	for i in range(0, 2000):
		model.w = [*map(lambda x:x*0, model.w)]
		np.random.shuffle(train_data)
		train_x = train_data[:, :4]
		train_y = train_data[:, 4]
		model.train(train_x, train_y)


		
		test_x = test_data[:, :4]
		b = np.asarray([1] * (test_x.shape[0]))
		test_x = np.vstack((b, test_x.transpose())).transpose()
		test_y = test_data[:, 4]
		s += model.loss(test_x, test_y, model.w)


	print(s/2000)
	# print(model.w)