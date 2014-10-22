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
thetaPosTrue = []
thetaNegTrue = []
#['!', '?', 'bad', 'beautiful', 'best', 'boring', 'great', 'love', 'still', 'stupid', 'superb', 'waste', 'wonderful', 'worst']
#dict = {('love','loving','loved','loves'): 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
#dict = {'love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst','stupid', 'waste', 'boring', '?', '!'}

dict = {'love': 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0,'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}


textDataSetsDirectoryFullPath = '/net/if24/jjl5sw/GitHub/ML_HW'
#xTrain, xTest, yTrain, yTest = loadData(textDataSetsDirectoryFullPath) 
def loadData(textDataSetsDirectoryFullPath):
	dict = {'love': 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
	xTrain = []
	for path, dirs, files in os.walk(textDataSetsDirectoryFullPath + 'training_set/pos'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			xTrain.append(transfer(fullpath, dict))

	for path, dirs, files in os.walk(textDataSetsDirectoryFullPath +'training_set/neg'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			xTrain.append(transfer(fullpath, dict))

	yTrain = []
	for i in range(0,700):
		yTrain.append(1)
	for i in range(700,1400):
		yTrain.append(-1)

	xTest = []
	for path, dirs, files in os.walk(textDataSetsDirectoryFullPath + 'test_set/pos'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			xTest.append(transfer(fullpath, dict))

	for path, dirs, files in os.walk(textDataSetsDirectoryFullPath + 'test_set/neg'):
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
	nNeg = []; nPos = [];
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
	for r in xTest:
		posSum = 0
		negSum = 0
		i = 0
		for c in r:
			for k in range(0,c):
				posSum += (math.log(thetaPos[i]) + math.log(0.5))
				negSum += (math.log(thetaNeg[i]) + math.log(0.5))
			i+=1
		if posSum > negSum:
			yPredict.append(1)
		else:
			yPredict.append(-1)

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
				pPos.append(math.log(float(nPos[sorted(vocabulary.keys()).index(word)]) / sum(nPos)))
				pNeg.append(math.log(float(nNeg[sorted(vocabulary.keys()).index(word)]) / sum(nNeg)))
				i += 1

	if sum(pPos) > sum(pNeg):
		return 1
	else:
		return -1
	

def testDirect(testFileDirectoryFullPath):
	dict = {'love': 0, 'wonderful': 0, 'best' :0, 'great': 0, 'superb': 0, 'still': 0, 'beautiful': 0, 'bad': 0, 'worst': 0,'stupid': 0, 'waste': 0, 'boring': 0, '?': 0, '!': 0}
	yPredict = []
	k = 0
	accuracy = 0
	for path, dirs, files in os.walk(testFileDirectoryFullPath + 'pos'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			yPredict.append(testDirectOne(fullpath))
			if yPredict[k] == 1:
				accuracy += 1;
			k += 1
	for path, dirs, files in os.walk(testFileDirectoryFullPath + 'neg'):
		for filename in files:
			fullpath = os.path.join(path, filename)
			yPredict.append(testDirectOne(fullpath))
			if yPredict[k] == -1:
				accuracy += 1;
			k += 1

	accuracy = float(accuracy)/len(yPredict)
	return yPredict, accuracy

def train2(xTrain,yTrain):
	global nNeg; global nPos;
	nNeg = []; nPos = []
	for i in range(0,14):
		nPos.append(0)
		nNeg.append(0)
		for j in range(0,1400):
			if yTrain[j] == 1:
				if xTrain[j][i] > 0:
					nPos[i] += 1
			else:
				if xTrain[j][i] > 0:
					nNeg[i] += 1

	#For each word, create P(w_k | class j)
	PwcPos = []; PwcNeg = []
	for n_k in nPos:
		PwcPos.append(float(n_k +1)/(700 +2))

	for n_k in nNeg:
		PwcNeg.append(float(n_k +1)/(700 +2))

	#theta = P(w_k | class_j)*P(class_j)	
	global thetaPosTrue ; global thetaNegTrue;
	thetaPosTrue = []; thetaNegTrue = [];
	for w in PwcPos:
		thetaPosTrue.append(w*0.5)

	for w in PwcNeg:
		thetaNegTrue.append(w*0.5)


	return (thetaPosTrue, thetaNegTrue)

def test2(xTest,yTest):
	#argmax log P(class j) + sum(log(P(xi | class j)))
	yPredict = []
	for r in xTest:
		posSum = 0
		negSum = 0
		i = 0
		for c in r:
			for k in range(0,c):
				posSum += (math.log(thetaPosTrue[i]) + math.log(0.5))
				negSum += (math.log(thetaNegTrue[i]) + math.log(0.5))
			i+=1
		if posSum > negSum:
			yPredict.append(1)
		else:
			yPredict.append(-1)

	classSum = 0
	for i in range(0,len(yPredict)):
		if yPredict[i] == yTest[i]:
			classSum += 1

	accuracy = float(classSum)/len(yPredict)		

	return yPredict, accuracy

def debug():
	# textDataSetsDirectoryFullPath = '/net/if24/jjl5sw/GitHub/ML_HW/'
	textDataSetsDirectoryFullPath = '/Users/Jack/GitHub/ML_HW3/'
	testFileDirectoryFullPath = '/Users/Jack/GitHub/ML_HW3/test_set/'
	Xtrain, Xtest, ytrain, ytest = naiveBayesMulFeature.loadData(textDataSetsDirectoryFullPath)
	thetaPos, thetaNeg = naiveBayesMulFeature.train(Xtrain, ytrain)
	yPredict, Accuracy = naiveBayesMulFeature.test(Xtest, ytest)
	yPredict, Accuracy = naiveBayesMulFeature.testDirect(testFileDirectoryFullPath)
	thetaPosTrue, thetaNegTrue= naiveBayesMulFeature.train2(Xtrain, ytrain)
	yPredict, Accuracy= naiveBayesMulFeature.test2(Xtest, ytest)


