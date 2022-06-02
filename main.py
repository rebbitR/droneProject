
import classes
from log import write_to_log
from video import video_to_frames,frames_to_video
# from multiprocessing import Pool
# import threading
from datetime import datetime

def print_current_time():
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    write_to_log(current_time)

def main(path):

    buf1=video_to_frames(path)
    buf=[]
    for i in range(len(buf1)-1):
        frame=classes.Frame(buf1[i],i)
        buf.append(frame)
    write_to_log(str(len(buf))+" frames---")

    print_current_time()
    for frame in buf:
        # find_objects at frame _with_yolo:
        frame.detect_objects()

        # cut_objects_from_frame():
        frame.cut_objects()

        # find_kinds_with_model:

        # try multiprocessing
        # t = []
        # arr = ['resnet_50', 'binary_vgg16', 'category_vgg16']
        # for i in range(3):
        #     t.append(threading.Thread(target=frame.model_detect, args=(arr[i],)))
        #     t[i].start()
        # for i in range(3):
        #     t[i].join()

        print_current_time()
        # frame.model_detect('resnet_50')
        # frame.model_detect('binary_vgg16')
        frame.model_detect('category_vgg16')

        frame.result()
        frame.print_results_frame()
        print_current_time()
        frame.add_txt()
        # frame1.show_img()
        buf[frame.idC]=frame.frameC
    print_current_time()




    # buf1 = []
    # for i in buf:
    #     buf1.append(i.frameC)

    return buf

if __name__ == '__main__':

    buf = main("video/V_AIRPLANE_042.mp4")
    frames_to_video("airplane-res2.mp4",buf)





