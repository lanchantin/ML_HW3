#!/usr/bin/python
import sys
import numpy as np
import math
import random
import os, os.path



dict = {'love': 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
#dict = {('love','loving','loved','loves'): 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
#dict = {'love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst','stupid', 'waste', 'boring', '?', '!'}

for path, dirs, files in os.walk('training_set/'):
	for filename in files:
		fullpath = os.path.join(path, filename)
		f = open(fullpath)
		for line in f:
			for word in line.split():
				if word in dict:
					dict[word] += 1
n = []
for key in sorted(dict.keys()):
	n.append(dict[key])




def train(xTrain,yTrain):
	DIR = 'training_set/neg/'
	numNegatives = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

	DIR = 'training_set/pos/'
	numPositives = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

	P_pos = float(numPositives)/(numPositives + numNegatives)
	P_neg = float(numNegatives)/(numPositives + numNegatives)






def transfer(fileDj,vocabulary):
	f = open(fileDj)
	n = []
	for i in range(0,len(vocabulary.keys())):
		n.append(0)
	for line in f:
		for word in line.split():
			if word in vocabulary:
				print word
				print sorted(dict.keys()).index(word)
				n[sorted(dict.keys()).index(word)] += 1
	return n
