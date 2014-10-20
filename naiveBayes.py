#!/usr/bin/python
import sys
import numpy as np
import math
import random
import os, os.path
from sklearn.naive_bayes import MultinomialNB


nNeg = []
nPos = []
thetaPos = []
thetaNeg = []
#['!', '?', 'bad', 'beautiful', 'best', 'boring', 'great', 'love', 'still', 'stupid', 'superb', 'waste', 'wonderful', 'worst']
#dict = {('love','loving','loved','loves'): 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
#dict = {'love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst','stupid', 'waste', 'boring', '?', '!'}

dict = {'love': 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0,'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}


textDataSetsDirectoryFullPath = '/net/if24/jjl5sw/GitHub/ML_HW'
#xTrain, xTest, yTrain, yTest = loadData(textDataSetsDirectoryFullPath) 
def loadData(textDataSetsDirectoryFullPath):
	dict = {'love': 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
	xTrain = []
	for path, dirs, files in os.walk(textDataSetsDirectoryFullPath + '/training_set/pos'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			xTrain.append(transfer(fullpath, dict))

	for path, dirs, files in os.walk(textDataSetsDirectoryFullPath +'/training_set/neg'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			xTrain.append(transfer(fullpath, dict))

	yTrain = []
	for i in range(0,700):
		yTrain.append(1)
	for i in range(700,1400):
		yTrain.append(-1)

	xTest = []
	for path, dirs, files in os.walk(textDataSetsDirectoryFullPath + '/test_set/pos'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			xTest.append(transfer(fullpath, dict))

	for path, dirs, files in os.walk(textDataSetsDirectoryFullPath + '/test_set/neg'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			xTest.append(transfer(fullpath, dict))

	yTest = []
	for i in range(0,len(xTest)/2):
		yTest.append(1)
	for i in range(len(xTest)/2,len(xTest)):
		yTest.append(-1)

	return (xTrain, xTest, yTrain, yTest)

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

def train(xTrain,yTrain):
	#create n_k = # of occurance of each word in all {pos, neg} texts
	global nNeg; global nPos;
	for i in range(0,14):
		nPos.append(0)
		nNeg.append(0)
		for j in range(0,1400):
			if yTrain[j] == 1:
				nPos[i] += xTrain[j][i]
			else:
				nNeg[i] += xTrain[j][i]

	#For each word, create P(w_k | class j)
	PwcPos = []; PwcNeg = []
	for n_k in nPos:
		PwcPos.append(float(n_k +1)/(sum(nPos) +14))

	for n_k in nNeg:
		PwcNeg.append(float(n_k +1)/(sum(nNeg) +14))

	#theta = P(w_k | class_j)*P(class_j)	
	global thetaPos; global thetaNeg;
	for w in PwcPos:
		thetaPos.append(w*0.5)

	for w in PwcNeg:
		thetaNeg.append(w*0.5)


	return (thetaPos, thetaNeg)

def test(xTest,yTest):
	#argmax log P(class j) + sum(log(P(xi | class j)))
	yPredict = []
	e = 0
	for r in xTest:
		posSum = 0
		negSum = 0
		i = 0
		for c in r:
			for k in range(0,c):
				posSum += math.log(thetaPos[i])
				negSum += math.log(thetaNeg[i])
			i+=1
		yPredict.append(0)
		if posSum > negSum:
			yPredict[e] = 1
		else:
			yPredict[e] = -1
		e += 1

	classSum = 0
	for i in range(0,len(yPredict)):
		if yPredict[i] == yTest[i]:
			classSum += 1

	accuracy = float(classSum)/len(yPredict)		

	return yPredict, accuracy


def testDirectOne(XtestTextFileNameInFullPathOne):
	vocabulary = {'love': 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
	f = open(XtestTextFileNameInFullPathOne)
	pNeg = []
	pPos = []
	i = 0
	for line in f:
		for word in line.split():
			if word in vocabulary:
				pPos.append(0)
				pNeg.append(0)
				pPos[i] = math.log(float(nPos[sorted(vocabulary.keys()).index(word)]) / sum(nPos))
				pNeg[i] = math.log(float(nNeg[sorted(vocabulary.keys()).index(word)]) / sum(nNeg))
				i += 1
	print sum(pPos)
	print sum(pNeg)

	if sum(pPos) > sum(pNeg):
		return 1
	else:
		return -1



def debug():
	textDataSetsDirectoryFullPath = '/net/if24/jjl5sw/GitHub/ML_HW/'
	Xtrain, Xtest, ytrain, ytest = naiveBayesMulFeature.loadData(textDataSetsDirectoryFullPath)
	thetaPos, thetaNeg = naiveBayesMulFeature.train(Xtrain, ytrain)
	yPredict, Accuracy = naiveBayesMulFeature.test(Xtest, ytest)

#def testDirect(testFileDirectoryFullPath):


# clf = MultinomialNB()
# clf.fit(xTrain, yTrain)
# clf.score(xTrain,yTrain)

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


