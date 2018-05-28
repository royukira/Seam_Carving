import cv2

img = cv2.imread("image/test6.jpg")

output = cv2.resize(img, (300, 375), interpolation=cv2.INTER_CUBIC)

cv2.imwrite("image/ns_test6.jpg", output)