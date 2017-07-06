import numpy as np
import math

def loadSimpData ():
	datMat = np.matrix(
	[[1., 2.1],
	[2., 1.1],
	[1.3, 1.1],
	[1., 1.1],
	[2., 1.]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return datMat, classLabels

def stumpClassify (dataMatrix, dimen, threshVal, threshIneq):
	retArray = np.ones((np.shape(dataMatrix)[0], 1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:, dimen] <= threshVal] = -1.0 #截取矩阵dimen列的所有元素,满足条件的全部赋值为-1
	else:
		retArray[dataMatrix[:, dimen] > threshVal] = 1.0
	return retArray

def buildStump (dataArray, classLabels, D):
	dataMatrix = np.mat(dataArray)
	labelMatrix = np.mat(classLabels).T
	m, n = np.shape(dataMatrix)
	numSteps = 10.0
	bestStump = {}
	bestClassEst = np.mat(np.zeros((m, 1)))
	minError = 0.3
	for i in range(n):
		rangeMin = dataMatrix[:, i].min()
		rangeMax = dataMatrix[:, i].max()
		stepSize = (rangeMax - rangeMin) / numSteps
		for j in range(-1, int(numSteps) + 1):
			for inquel in ['lt', 'gt']:
				threshVal = (rangeMin + float(j) * stepSize)
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inquel)
				errArr = np.mat(np.ones((m ,1)))
				errArr[predictedVals == labelMatrix] = 0
				weightedError = D.T * errArr
				if weightedError < minError:
					minError = weightedError
					bestClassEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inquel
	return bestStump, minError, bestClassEst

def adaBoostTrainDS (dataArray, classLabels, numIt = 40):
	weakClassArr = []
	m = np.shape(dataArray)[0]
	D = np.mat(np.ones((m, 1)) / m)
	aggClassEst = np.mat(np.ones((m, 1)) / m)
	for i in range(numIt):
		bestStump, error, classEst = buildStump(dataArray, classLabels, D)
		print ("D:", D.T)
		alpha = np.float(0.5 * math.log((1.0 - error) / max(error, 1e-16))) #计算这个分类器的话语权
		bestStump['alpha']	= alpha
		weakClassArr.append(bestStump)
		print ("classEst：", classEst.T)
		expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
		D = np.multiply(D, np.exp(expon))
		D = D / D.sum()
		aggClassEst += alpha * classEst
		print ("aggClassEst:", aggClassEst.T)
		aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
		errorRate = aggErrors.sum() / m
		print ("total error: ", errorRate)
		if errorRate == 0.0: break
	return weakClassArr

def adaClassify (dataToClass, classifierArray):
	dataMatrix = np.mat(dataToClass)
	m = np.shape(dataMatrix)[0]
	aggClassEst = np.mat(np.zeros((m, 1)))
	for i in range(len(classifierArray)):
		classEst = stumpClassify(dataMatrix, classifierArray[i]['dim'],\
		classifierArray[i]['thresh'],\
		classifierArray[i]['ineq'])
		aggClassEst += classifierArray[i]['alpha'] * classEst
		print (aggClassEst, classifierArray[i]['alpha'], classEst)
	return np.sign(aggClassEst)

datMat, classLabels = loadSimpData()
print("dataMat:", datMat, "classLabels:", classLabels)

classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
print("classifierArray:", classifierArray)

result = adaClassify([1.2, 2], classifierArray)
print("result:", result)
