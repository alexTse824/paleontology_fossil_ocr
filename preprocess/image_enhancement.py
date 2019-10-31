import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = '/Users/xie/Code/paleontology_fossil_ocr/data/preprocess_data/image_enhancement/4.jpg'


def grubcut(image_path):
    img = cv2.imread(image_path)
    OLD_IMG = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    SIZE = (1, 65)
    bgdModle = np.zeros(SIZE, np.float64)
    fgdModle = np.zeros(SIZE, np.float64)
    rect = (1, 1, img.shape[1], img.shape[0])
    cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img *= mask2[:, :, np.newaxis]

    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("grabcut"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(OLD_IMG, cv2.COLOR_BGR2RGB))
    plt.title("original"), plt.xticks([]), plt.yticks([])

    plt.show()


def get_outer(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 转换为二值图像
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 提取轮廓
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    print('ready to show')
    cv2.imshow("img", img)
    cv2.waitKey(0)


# grubcut(image_path)

img = cv2.imread(image_path)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#绘制独立轮廓，如第四个轮廓
#imag = cv2.drawContour(img,contours,-1,(0,255,0),3)
#但是大多数时候，下面方法更有用
imag = cv2.drawContours(img,contours,3,(0,255,0),3)

while(1):
    cv2.imshow('img',img)
    cv2.imshow('imgray',imgray)
    cv2.imshow('image',image)
    cv2.imshow('imag',imag)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()