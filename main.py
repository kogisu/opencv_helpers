import cv2
import numpy as np
import matplotlib.pyplot as plt
from templateMatch import templateMatch

def showImage(img):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('gray', image)
    k = cv2.waitKey(0) & 0xFF
    if (k == 'q'):
        cv2.destroyAllWindows()

# showImage('parrot.jpg')



def maskVideoRed():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([170, 70, 90])
        upper_red = np.array([179, 255, 255])
        mask = cv2.inRange(gray_hsv, lower_red, upper_red)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('redout', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# maskVideoRed()

def modifyPixels2D(img, x, y, new):
    img[x,y] = new
    print (img[x, y])

def createLine():
    img = cv2.imread('parrot.jpg', cv2.IMREAD_COLOR)
    lineThickness = 2
    cv2.line(img, (0, 0), (200, 200), (0, 255, 0), 2)
    cv2.imshow('crossed image', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

# createLine()

def weightImages(img1, img2, w1, w2):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)
    gamma = 0

    weighted = cv2.addWeighted(image1, w1, image2, w2, gamma)
    cv2.imshow('weighted', weighted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def threshold(img):
    image = cv2.imread(img)
    greyscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, threshold = cv2.threshold(image, 12, 255, cv2.THRESH_BINARY)
    # ret2, threshold2 = cv2.threshold(greyscaled, 12, 255, cv2.THRESH_BINARY)
    gaussian = cv2.adaptiveThreshold(greyscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # ret3, otsu = cv2.threshold(greyscaled, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('image', image)
    cv2.imshow('gaussian', gaussian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# threshold('bookpage.jpg')

def filterColor():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([150, 150, 150])
        upper_red = np.array([250, 250, 255])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        #Add blurring
        # kernel = np.ones((15, 15), np.float32) / 225
        # smoothed = cv2.filter2D(res, -1, kernel)
        # gaussian = cv2.GaussianBlur(res, (15, 15), 0)
        medianBlur = cv2.medianBlur(res, 15)

        #Add morphological transformation
        kernel = np.ones((5, 5), np.uint(8))
        erosion = cv2.erode(mask, kernel, iterations = 1)
        dilation = cv2.dilate(mask, kernel, iterations = 1)
        # cv2.imshow('smoothed', medianBlur)
        # cv2.imshow('erosion', erosion)
        # cv2.imshow('dilation', dilation)

        #opening - removing false positives (in the background)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('opening', opening)

        #closing - removing false negatives (in the object)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('closing', closing)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

# filterColor()

templateMatch()
