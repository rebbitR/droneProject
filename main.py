# main number 2 try option 2 with classes_try:---------

from log import getLog,send_buf_to_log
# from cut_video import cut_video_to_frame
# import cv2
#
# def program(path):
#     buf=cut_video_to_frame(path)
#     for frame1 in buf:
#         # find_objects at frame _with_yolo:
#         frame1.yolo_detect()
#         # cut_objects_from_frame():
#         frame1.cut_objects()
#         # find_kinds_with_model:
#         frame1.model()
#         frame1.result()
#         cv2.imwrite("hhhhhhhhhh.png", frame1.frameC)
#     # send_buf_to_log(buf)
#     # log = getLog()
#     return buf
#
#
#
# buf = program("video/V_AIRPLANE_001.mp4")



# from classes import try1
# one=try1(1)
# two=try1(2)
# three=try1(3)
# print(one.x)
# buf=[]
# buf.append(one)
# buf.append(two)
# buf.append(three)
# for i in buf:
#     i.replace()
# for i in buf:
#     print(i.x)

# from cut_video import cut_video_to_frame
import cv2
from classes import frame

def program(path):
    img=cv2.imread(path)
    # cv2.imshow('', img)
    # cv2.waitKey(500)
    # cv2.destroyAllWindows()
    frame1=frame(img,1)
    cv2.imshow('', frame1.frameC)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    # find_objects at frame _with_yolo:
    frame1.yolo_detect()
    # cut_objects_from_frame():
    frame1.cut_objects()
    # find_kinds_with_model:
    frame1.model()
    frame1.result()
    cv2.imwrite("result.png", frame1.frameC)
    result=cv2.imread("result.png")
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # send_buf_to_log(buf)
    # log = getLog()




buf = program("images/birds-flying-overcast-sky-12614779.jpg")

