import cv2
import numpy as np

def templateMatch():
    img_bgr = cv2.imread('opencv-template-matching-python-tutorial.jpg')
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img_bgr', img_bgr)

    template = cv2.imread('opencv-template-for-matching.jpg', 0)
    print (template.shape)

    #template returns flipped size (y, x)
    w, h = template.shape[:: -1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.70
    loc = np.where(res >= threshold)

    #zip iterates over array as if they were arguments... this generates arguments of x,y pairs
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_bgr, pt, (pt[0]+w, pt[1]+h), (0, 255, 255), 2)

    cv2.imshow('detected', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

