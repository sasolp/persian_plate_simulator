
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("D:\\ANPR_yooztaa\\TestingData\\sh\\kia_sh.jpg")
thresh = 90

cropImage = img[579 : 612 ,1720  : 1757 , 0]
# cropImage2 = img[7 : 36 , 48 : 76]
# imageBinary = img > thresh
imageBinary = cropImage.copy()
cv2.threshold(cropImage, thresh, 255, cv2.THRESH_BINARY, imageBinary)
cv2.imwrite("D:\\ANPR_yooztaa\\TestingData\\sh\\sh_binary_2.bmp" , imageBinary)

plt.imshow(imageBinary , cmap=plt.cm.gray)
plt.show()
