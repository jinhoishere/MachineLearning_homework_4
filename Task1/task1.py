import cv2

img_gray = cv2.imread('CAT.jpg', 0)
img_original = cv2.imread('CAT.jpg')

cv2.imshow('cat_gray', img_gray)
cv2.imshow('cat_original', img_original)
cv2.waitKey(0)
cv2.destroyAllWindows
