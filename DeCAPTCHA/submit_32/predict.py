# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.

import cv2
import pickle
import numpy as np

def bgclear(img):
    # BG Remover 3
    p = img[0][0]
    img = img - p
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    kernel = np.ones((10, 10), np.uint8)
    img = cv2.erode(img, kernel,iterations=1)
    return img

def removeSpaces(img):
    l = []
    for i in img:
        if(np.sum([i])!=0):
            l.append(i)
    tem_np = np.array(l)
    l = []
    for i in tem_np.transpose():
        if(np.sum([i])!=0):
            l.append(i)
    tem = np.array(l)
    tem = tem.transpose()
    return tem_np

def splitImages(img):
    c = 0
    l = []
    j = np.zeros(img.shape[1])
    for i in img.transpose():
        if(np.sum(j)==0 and np.sum(i)!=0):
            l.append(c)
        if(np.sum(j)!=0 and np.sum(i)==0):
            l.append(c)
        j = i
        c+=1
    return l


def decaptcha( filenames ):
	# The use of a model file is just for sake of illustration

	dic = {1:'ALPHA',2:'BETA',3:'CHI',4:'DELTA',5:'EPSILON',6:'ETA',7:'GAMMA',8:'IOTA',9:'KAPPA',10:'LAMDA',11:'MU',12:'NU',13:'OMEGA',14:'OMICRON',15:'PHI',16:'PI',17:'PSI',18:'RHO',19:'SIGMA',20:'TAU',21:'THETA',22:'UPSILON',23:'XI',24:'ZETA'}
	
	data = []
	for src in filenames:
		img = cv2.imread(src)
		img = bgclear(img)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		arr = splitImages(img)
		if(len(arr) != 6):
			img1 = img[:,0:170]
			img2 = img[:,170:350]
			img3 = img[:,350:500]
		else:
			img1 = img[:, max(0, arr[0]-5):arr[1]+5]
			img2 = img[:, arr[2]-5:arr[3]+5]
			img3 = img[:, arr[4]-5:min(arr[5]+5,499)]
		img1 = removeSpaces(img1)
		img2 = removeSpaces(img2)
		img3 = removeSpaces(img3)
		img1 = cv2.resize(img1, (50, 50), interpolation = cv2.INTER_AREA)
		img2 = cv2.resize(img2, (50, 50), interpolation = cv2.INTER_AREA)
		img3 = cv2.resize(img3, (50, 50), interpolation = cv2.INTER_AREA)
		data.append(img1.flatten())
		data.append(img2.flatten())
		data.append(img3.flatten())

	X = np.array(data)

	model = pickle.load(open('model.sav', 'rb'))
	pred = model.predict(X)
	labels = []
	for i in range(len(filenames)):
		tem = dic[pred[3*i+0]]+','+dic[pred[3*i+1]]+','+dic[pred[3*i+2]]
		labels.append(tem)

	return labels