
import cv2

def try_cut_nparray(path,x,y,w,h):
    img=cv2.imread(path)
    cv2.imshow('after', img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    crop_img=img[y:y+h,x:x+w]
    cv2.imshow('after', crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

try_cut_nparray('images/drone6.jpg',166,97,364,180)