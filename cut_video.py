
import cv2
# from os import makedirs
# from os.path import splitext, dirname, basename, join
import classes



def cut_video_to_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return
    # # to save the images:
    # v_name = splitext(basename(video_path))[0]
    # video_path_arr= video_path.split('/')
    # d_name=video_path_arr[-2]
    # d1_name = video_path_arr[-3]
    # print(video_path_arr)
    # if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
    #     frame_dir = dirname(frame_dir)
    # frame_dir_ = join(frame_dir,d1_name,d_name, v_name)
    # makedirs(frame_dir_, exist_ok=True)
    # base_path = join(frame_dir_, name)
    buf=[]

    idx = 0
    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()
        if ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  #Save 0 second frame
                # cv2.imwrite("{}_{}.{}".format(base_path, "0000", ext),
                #             frame)
                second = 0
                myf1 = classes.frame(frame, second)
                buf.append(myf1)


            elif idx < cap.get(cv2.CAP_PROP_FPS):
                continue
            else:  #Save frames 1 second at a time
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                # print(second)
                # filled_second = str(second).zfill(4)
                # print(filled_second)
                # print(base_path)
                # print(ext)
                # print(idx)
                myf1 =classes.frame(frame, second)
                # cv2.imwrite("{}_{}.{}".format(base_path, filled_second, ext),
                #             frame)
                buf.append(myf1)
                idx = 0
        else:
            break

    return buf


# # try
#
# buf=cut_video_to_frame("video/try/sea.avi")
# # print(type(buf[0].frameC))
# for frame in buf:
#     cv2.imshow("cropped", frame.frameC)
#     print(type(frame.frameC))
#     cv2.waitKey(500)
#     cv2.destroyAllWindows()




