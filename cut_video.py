
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
                # cv2.imshow('',frame)
                # cv2.waitKey(500)
                # cv2.destroyAllWindows()

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
                # cv2.imshow('',frame)
                # cv2.waitKey(500)
                # cv2.destroyAllWindows()

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
# buf=cut_video_to_frame("")
# print(type(buf[0].frameC))
# mone=0
# for frame in buf:
#     print(type(frame.frameC))
#     cv2.imshow("cropped", frame.frameC)
#
#     cv2.waitKey(50)
#     cv2.destroyAllWindows()
#     mone = mone + 1
# print(mone)



def videotoframes(video):
 vidcap = cv2.VideoCapture(video)
#שמירת השמע
 #clip = mp.VideoFileClip(video).subclip(0, 20)
 #clip.audio.write_audiofile(r"my_result.mp3")
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


import cv2
import numpy as np
import glob

def test2(list=[]):
    height, width, layers = list[0].shape
    frameSize = (500, 500)
    print(height, width)
    out = cv2.VideoWriter('output3_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

    for img in list:
        out.write(img)

    # for i in range(0, 255):
    #     img = np.ones((500, 500, 3), dtype=np.uint8) * i
    #     out.write(img)

    # cv2.destroyAllWindows()
    out.release()

from dataset import changeResolution2
def cut_place(image,place,size):
    # print(place)
    x=round(float(place[0]))
    y=round(float(place[1]))
    w = round(float(place[2]))
    h = round(float(place[3]))
    # count=h*w
    # print(count, "pixels")
    buf_img=[]
    if(h>size or w>size):
        if(h<size):
            h=size
        elif(w<size):
            w=size
        crop_img = image[y:y + h, x:x + w]
        crop_img=changeResolution2(crop_img,size)

    else:
        w=size
        h=size
        crop_img = image[y:y + h, x:x + w]


    return crop_img

# list=videotoframes('D:/קבצים גדולים פרוייקט רננה/video/V_AIRPLANE_048')
# newList=[]
# for i in list:
#     height, width, layers=i.shape
#     print(height,width)
#     # cv2.imshow('before', i)
#     # cv2.waitKey(100)
#     # cv2.destroyAllWindows()
#     newi=cut_place(i,[0,0,0,0],500)
#     height, width, layers=newi.shape
#     print(height,width)
#     # cv2.imshow('after', newi)
#     # cv2.waitKey(100)
#     # cv2.destroyAllWindows()
#     newList.append(newi)
#
# # for i in newList:
# #     cv2.imshow('txt', i)
# #     cv2.waitKey(100)
# #     cv2.destroyAllWindows()
# test2(list)




