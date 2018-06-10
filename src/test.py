from skin_color_classifier import SkinColorClassifier as SCC
import cv2


in_dir = "../test-data/"
img1 = cv2.imread(in_dir + "img_max.png")
img2 = cv2.imread(in_dir + "img_min.png")
test = cv2.imread(in_dir + "img_test.png")

scc = SCC(img1, img2)
res = scc.mask_image(test)
cv2.imwrite(in_dir + "res.png", res)


