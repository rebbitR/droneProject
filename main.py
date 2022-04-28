# main number 2 try option 2 with classes_try:---------

from log import getLog,send_buf_to_log
from cut_video import cut_video_to_frame
import cv2

def program(path):
    buf=cut_video_to_frame(path)
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
    return buf

buf = program("video/V_BIRD_016.mp4")
for i in buf:
    i.show_img(txt='result',waitKey=0)
# buf_2=[]
# for i in buf:
#     buf_2.append(i.frameC)
#
# framestovideo(buf_2)




