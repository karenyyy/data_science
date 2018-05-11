#!/usr/bin/env python
#largely based on https://realpython.com/blog/python/face-recognition-with-python/
import cv2
import numpy as np
from urllib.request import urlopen
import glob

def readImage(imgPath):
    if 'http' in imgPath:
        try:
            req = urlopen(imgPath)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image = cv2.imdecode(arr,-1)
        except:
            return None
    else:
        image = cv2.imread(imgPath)
    return image

def detectFaces(imgPath, cascPath):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = readImage(imgPath)
    if image is None:
        print("Error reading %s" % imgPath)
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

    print("Found %d faces in %s!" % (len(faces), imgPath))
    return len(faces)

def facesAreSimilar(imgPath1, imgPath2):
    image1 = cv2.cvtColor(readImage(imgPath1), cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(readImage(imgPath2), cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([image1.astype('float32')], channels=[0], mask=None, histSize=[64], ranges=[0,64])
    hist2 = cv2.calcHist([image2.astype('float32')], channels=[0], mask=None, histSize=[64], ranges=[0,64])

    similarity = cv2.compareHist(hist1, hist2, method = cv2.HISTCMP_CORREL)
    print("Similarity score %s vs %s: %f.2" % (imgPath1, imgPath2, similarity))
    return similarity > 0.98

