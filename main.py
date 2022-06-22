
import classes
from log import write_to_log
from video import video_to_frames,frames_to_video,cut_video_by_second
# from multiprocessing import Pool
# import threading
# from datetime import datetime

# def print_current_time():
#     now = datetime.now()
#
#     current_time = now.strftime("%H:%M:%S")
#     print(current_time)

def main(path):

    buf_frames=video_to_frames(path)
    print('num of frames: '+str(len(buf_frames)))
    buf=[]
    for i in range(len(buf_frames)):
        frame=classes.Frame(buf_frames[i],i)
        buf.append(frame)
    write_to_log(str(len(buf))+"New Video:-----")
    # print(print_current_time())
    for frame in buf:

        # find objects at frame  with yolo:
        frame.detect_objects()

        # cut objects from frame:
        frame.cut_objects()

        # find_kinds_with_model:
        # frame.model_detect('category_vgg16')
        # frame.model_detect('binary_vgg16')
        frame.model_detect('resnet_50')

        # # try multiprocessing
        # t = []
        # arr = ['binary_vgg16', 'category_vgg16']
        # for i in range(2):
        #     t.append(threading.Thread(target=frame.model_detect, args=(arr[i],)))
        #     t[i].start()
        # for i in range(2):
        #     t[i].join()


        # result:
        frame.result()

        # print to log the results of this frame:
        frame.print_results_frame()

        # add txt to the objects in the frame:
        frame.add_txt()

        # show the frame result:
        # frame.show_img()

        buf[frame.id]=frame.frame

    # print(print_current_time())

    return buf







