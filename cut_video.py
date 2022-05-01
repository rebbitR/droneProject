
import cv2
import classes

# from os import makedirs
# from os.path import splitext, dirname, basename, join

def cut_video_by_second(video_path: str):

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

                myf1 =classes.frame(frame, second)
                # cv2.imwrite("{}_{}.{}".format(base_path, filled_second, ext),
                #             frame)
                buf.append(myf1)
                idx = 0
        else:
            break

    return buf

def videotoframes(video):
 vidcap = cv2.VideoCapture(video)

 list=[]
 def getFrame(sec):
   vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
   hasFrames,image = vidcap.read()

   if hasFrames:
    list.append(image)

    #cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
 sec = 0
 frameRate = 0.08 #//it will capture image in each 0.5 second
 count=1
 success = getFrame(sec)
 while success:
   count = count + 1
   sec = sec + frameRate
   sec = round(sec, 2)
   success = getFrame(sec)
 return list

def framestovideo(video_name,list=[]):
    height, width, layers = list[0].shape
    frameSize = (width,height)
    # print(height, width)
    out = cv2.VideoWriter(video_name, 0, 12, frameSize)
    for img in list:
        out.write(img)

    cv2.destroyAllWindows()
    out.release()


# list=videotoframes('D:/droneProject111/video/V_BIRD_005.mp4')





