
import numpy as np
import cv2

def VideoToImage(video):
    cap=cv2.VideoCapture(video)

    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frameCount=frame_count // 12
    buf=np.empty((frameCount,frame_width,frame_height,3),np.dtype('uint8'))

    fc=0
    ret=True
    buf2=[]
    while fc<frameCount and ret:
        ret,buf[fc]=cap.read()
        fc+=1

    for i in range(0,len(buf),12):
        buf2.append(buf[i])
    cap.release()

    cv2.namedWindow('frame 10')
    cv2.imshow('frame 10',buf[9])

    cv2.waitKey(0)
    print(len(buf))
    print(frameCount)
    return buf

# list=VideoToImage("video/sea.avi")