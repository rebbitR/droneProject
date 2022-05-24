# main number 2 try option 2 with classes_try:---------
import threading

import classes
from log import getLog,send_buf_to_log
from cut_video import cut_video_by_second,videotoframes,framestovideo
from load_model import my_model
import cv2
from multiprocessing import Pool

from datetime import datetime


def main(path):

    # buf=cut_video_by_second(path)
    buf1=videotoframes(path)
    buf=[]
    for i in range(len(buf1)-1):
        frame=classes.frame(buf1[i],i)
        buf.append(frame)
        # print(type(buf[i]))
    print(len(buf))

    # now = datetime.now()
    #
    # current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)
    # buf[111].yolo_detect()
    #
    # # cut_objects_from_frame():
    # buf[111].cut_objects()
    # # find_kinds_with_model:
    # buf[111].model('resnet_50')
    # # buf[111].model('binary_vgg16')
    # # buf[111].model('category_vgg16')
    # buf[111].result()
    # # now = datetime.now()
    # #
    # # current_time = now.strftime("%H:%M:%S")
    # # print("Current Time =", current_time)
    # buf[111].print_results_frame()
    # buf[111].tryAddTxt()
    # now = datetime.now()
    #
    # current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    for frame1 in buf:
        # find_objects at frame _with_yolo:
        frame1.yolo_detect()

        # cut_objects_from_frame():
        frame1.cut_objects()
        # find_kinds_with_model:
        # try multiprocessing
        # t=[]
        # arr=['resnet_50', 'binary_vgg16']
        # for i in range(2):
        #     t.append(threading.Thread(target=frame1.model,args=(arr[i],)))
        #     t[i].start()
        # for i in range(2):
        #     t[i].join()
        t = []
        arr = ['resnet_50', 'binary_vgg16', 'category_vgg16']
        for i in range(3):
            t.append(threading.Thread(target=my_model, args=('object0.png', 81, arr[i])))
            t[i].start()
        for i in range(3):
            t[i].join()
        # with Pool(3) as p:
        #     p.map(frame1.model, ['resnet_50', 'binary_vgg16', 'category_vgg16'])
        # frame1.model('resnet_50')
        # frame1.model('binary_vgg16')
        # frame1.model('category_vgg16')
        frame1.result()
        frame1.print_results_frame()
        frame1.tryAddTxt()
        # frame1.show_img()
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)



    # send_buf_to_log(buf)
    # log = getLog()



    return buf

if __name__ == '__main__':

    buf = main("video/V_AIRPLANE_048.mp4")
# for i in buf:
#     i.show_img()

# buf1=[]
# for i in buf:
#     buf1.append(i.frameC)
# framestovideo("airplane-res.mp4",buf1)





