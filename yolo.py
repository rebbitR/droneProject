import cv2
import numpy as np
from PIL import Image
import os
import PIL
#
#
#
# def yolo_detect1(list_pic):
#     classNames=[]
#     pre_img=[]
#     classFile='coco.names'
#     with open(classFile,'rt') as f:
#         classNames=f.read().rstrip('\n').split('\n')
#     configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
#     weightsPath='frozen_inference_graph.pb'
#     net = cv2.dnn_DetectionModel(weightsPath,configPath)
#     net.setInputSize(320,320)
#     net.setInputScale(1.0/127.5)
#     net.setInputMean((127.5,127.5,127.5))
#     net.setInputSwapRB(True)
#     for pic in list_pic:
#         # cv2.dnn_DetectionModel.detect()
#         classIds,confs,bbox=net.detect(np.asarray(pic),0.5)
#         print(classIds)
#         print(confs)
#         print(bbox)
#         pre_img.append((pic,False))
#         print(len(classIds))
#         if len(classIds)!=0:
#             for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
#                 left=box[0]
#                 top=box[1]
#                 right=box[2]+box[0]
#                 bottom=box[3]+box[1]
#                 print(left,top,right,bottom)
#                 list=[]
#                 list.append(left)
#                 list.append(top)
#                 list.append(right)
#                 list.append(bottom)
#                 crop_image_from_yolo()
#
#
#     return pre_img
#
# def crop_image_from_yolo(name_of_image,place,save):
#     im = Image.open('lisbon.jpg').convert('L')
#     im = im.crop((left, top, right, bottom))
#     im.save('_2.png')
# img = cv2.imread("lisbon.jpg")
# list_pic=[]
# list_pic.append(img)
# pre_img=yolo_detect1(list_pic)
# # x,y,w,h
# # [y:y + h, x:x + w]

# img=cv2.imread('lisbon.jpg', 1)
# print(type(img))
# im = Image.open('lisbon.jpg').convert('L')
# print(type(im))

#
#
def yolo_detect1(list_pic):
    # classNames=[]
    pre_img=[]
    classFile='coco.names'

    with open(classFile,'rt') as f:
        classNames=f.read().rstrip('\n').split('\n')
    configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath='frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath,configPath)

    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)

    dir = 0
    for pic in list_pic:

        classIds,confs,bbox=net.detect(np.asarray(pic))
        pre_img.append((pic,False))

        numImg = 0
        if len(classIds)!=0:
            mylist = []

            for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):

                left=box[0]
                top=box[1]
                right=box[2]+box[0]
                bottom=box[3]+box[1]
                # print(type(pic))
                image1=Image.fromarray(pic)
                # print(type(image1))

                crop_image = image1.crop((left, top, right, bottom))

                # directory = str(dir)
                # # Parent Directory path
                # parent_dir = ""
                # # Path
                # path = os.path.join(parent_dir, directory)
                # os.mkdir(path)
                #
                # crop_image.save(directory+'/'+str(numImg)+'.png')

                crop_image.save(str(numImg)+classNames[classId-1]+'.png')
                np_img=np.array(crop_image)
                mylist.append(np_img)
                numImg=numImg+1
                # pre_img[-1]=(np_img,True)

            for pic in mylist:
                cv2.imshow('',pic)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
        print(numImg)
        dir=dir+1

    return pre_img

img = cv2.imread("lisbon.jpg")
list_pic=[]
list_pic.append(img)
pre_img=yolo_detect1(list_pic)
# pic=cv2.imread("lisbon.jpg")
# cv2.imshow('',pic)
# cv2.waitKey(500)
# cv2.destroyAllWindows()
