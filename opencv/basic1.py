import cv2

img = cv2.imread('piedpiper.png')

cv2.imwrite('compressed_pp.png', img, [cv2.IMWRITE_PNG_COMPRESSION])