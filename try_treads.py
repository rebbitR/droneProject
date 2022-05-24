import threading
from classes import frame
import cv2
from datetime import datetime

if __name__ == '__main__':
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    image=cv2.imread("video/V_AIRPLANE_0081_319.png")
    frame1=frame(image,1)
    frame1.yolo_detect()
    frame1.cut_objects()
    t=[]
    arr=['resnet_50', 'binary_vgg16']
    for i in range(2):
        t.append(threading.Thread(target=frame1.model,args=(arr[i],)))
        t[i].start()
    for i in range(2):
        t[i].join()
    # frame1.model('resnet_50')
    # frame1.model('binary_vgg16')
    # frame1.model('category_vgg16')
    frame1.result()
    frame1.print_results_frame()
    frame1.tryAddTxt()
    frame1.show_img()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)