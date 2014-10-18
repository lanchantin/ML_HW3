#!/usr/bin/python
import sys
import numpy as np
import math
import random
import os, os.path


#['!', '?', 'bad', 'beautiful', 'best', 'boring', 'great', 'love', 'still', 'stupid', 'superb', 'waste', 'wonderful', 'worst']
#dict = {('love','loving','loved','loves'): 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
#dict = {'love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst','stupid', 'waste', 'boring', '?', '!'}

dict = {'love': 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}





def train(xTrain,yTrain):


	nNeg = []
	nPos = []
	for i in range(0,14):
		nNeg.append(0)
		nPos.append(0)
		for j in range(0,1400):
			if yTrain[j] == 1:
				nPos[i] += xTrain[j][i]
			else:
				nNeg[i] += xTrain[j][i]

	#For each word, create P(w_k | class j)
	PwcPos = []
	for n_k in nPos:
		PwcPos.append(float(n_k +1)/(sum(nPos) +14))

	PwcNeg = []
	for n_k in nNeg:
		PwcNeg.append(float(n_k +1)/(sum(nNeg) +14))

	thetaPos = []
	for w in PwcPos:
		thetaPos.append(w*0.5)

	thetaNeg = []
	for w in PwcNeg:
		thetaNeg.append(w*0.5)

	return (thetaPos, thetaNeg)





def createMat():
	dict = {'love': 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
	xTrain = []
	for path, dirs, files in os.walk('training_set/pos'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			xTrain.append(transfer(fullpath, dict))

	for path, dirs, files in os.walk('training_set/neg'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			xTrain.append(transfer(fullpath, dict))

	yTrain = []
	for i in range(0,700):
		yTrain.append(1)
	for i in range(700,1400):
		yTrain.append(-1)

	return xTrain, yTrain



def transfer(fileDj,vocabulary):
	f = open(fileDj)
	n = []
	for i in range(0,len(vocabulary.keys())):
		n.append(0)
	for line in f:
		for word in line.split():
			if word in vocabulary:
				n[sorted(dict.keys()).index(word)] += 1
	return n



	# numNegatives = len([name for name in os.listdir('training_set/neg/') if os.path.isfile(os.path.join('training_set/neg/', name))])
	# numPositives = len([name for name in os.listdir('training_set/pos/') if os.path.isfile(os.path.join('training_set/pos/', name))])

	# P_pos = float(numPositives)/(numPositives + numNegatives)
	# P_neg = float(numNegatives)/(numPositives + numNegatives)


	# #Create n_k for each class
	# sets = ['training_set/pos','training_set/neg']
	# for set in sets:
	# 	dict = {'love': 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
	# 	for path, dirs, files in os.walk(set):
	# 		for filename in files:
	# 			fullpath = os.path.join(path, filename)
	# 			f = open(fullpath)
	# 			for line in f:
	# 				for word in line.split():
	# 					if word in dict:
	# 						dict[word] += 1
	# 	if set == 'training_set/pos':
	# 		nPos = []
	# 		for key in sorted(dict.keys()):
	# 			nPos.append(dict[key])
	# 	else:
	# 		nNeg = []
	# 		for key in sorted(dict.keys()):
	# 			nNeg.append(dict[key])