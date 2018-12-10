import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def featureMatch():
    img1 = cv2.imread('images/opencv-feature-matching-template.jpg', 0)
    img2 = cv2.imread('images/opencv-feature-matching-image.jpg', 0)


    orb = cv2.ORB_create()
    print ('orb: ', orb)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    print ('desc1: ', des1, 'desc2: ', des2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    plt.imshow(img3)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()