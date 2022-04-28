# main number 2 try option 2 with classes_try:---------
import classes
from log import getLog,send_buf_to_log
from cut_video import cut_video_to_frame,videotoframes

import time

import cv2

def program(path):
    t1=time.time()
    buf=cut_video_to_frame(path)
    # buf1=videotoframes(path)
    # buf=[]
    # for i in range(len(buf1)-1):
    #     frame=classes.frame(buf1[i],i)
    #     buf.append(frame)
    #     print(type(buf[i]))
    # print(len(buf))
    for frame1 in buf:
        # find_objects at frame _with_yolo:
        frame1.yolo_detect()

        # cut_objects_from_frame():
        frame1.cut_objects()
        # find_kinds_with_model:
        frame1.model('resnet_50')
        frame1.model('binary_vgg16')
        frame1.model('category_vgg16')
        frame1.result()
        frame1.print_results_frame()
        # frame1.show_img()
        # if len(frame1.objectsC)!=0:
        #     for my_obg in frame1.objectsC:
        #         print(my_obg.models)
        # frame1.show_img(txt='its worked!!!',waitKey=0)



    # send_buf_to_log(buf)
    # log = getLog()

    t2=time.time()
    print('time s: ',t1-t2)

    return buf

buf = program("video/V_DRONE_027.mp4")
for i in buf:
    i.show_img(txt='result',waitKey=0)
# buf_2=[]
# for i in buf:
#     buf_2.append(i.frameC)
#
# framestovideo(buf_2)




