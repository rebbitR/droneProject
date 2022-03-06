# main number 2 try option 2 with classes_try:---------

from log import getLog,send_buf_to_log
from cut_video import cut_video_to_frame


def program(path):
    buf=cut_video_to_frame(path)
    for frame1 in buf:
        frame1.find_objects_with_yolo()
        frame1.cut_objects_from_frame()
        frame1.find_kinds_with_model()
        frame1.result()
    send_buf_to_log(buf)
    log = getLog()
    return log



# log = program("video/V_AIRPLANE_001.mp4")
