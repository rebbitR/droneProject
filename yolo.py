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
#
# def yolo_detect_return_dirs(list_pic):
#
#     pre_img=[]
#     classFile='yolo_file/coco.names'
#
#     with open(classFile,'rt') as f:
#         classNames=f.read().rstrip('\n').split('\n')
#     configPath='yolo_file/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
#     weightsPath='yolo_file/frozen_inference_graph.pb'
#     net = cv2.dnn_DetectionModel(weightsPath,configPath)
#
#     net.setInputSize(320,320)
#     net.setInputScale(1.0/127.5)
#     net.setInputMean((127.5,127.5,127.5))
#     net.setInputSwapRB(True)
#
#     list1=[]
#     dir = 0
#     for pic in list_pic:
#
#         # print(list_pic[dir])
#
#         classIds,confs,bbox=net.detect(np.asarray(pic))
#         pre_img.append((pic,False))
#         numImg = 0
#
#         if len(classIds)!=0:
#             list2 = []
#
#             # Parent Directory path
#             parent_dir = "yolo_detect"
#             # Path
#             path = os.path.join(parent_dir, str(dir)+'_'+'pic')
#             os.mkdir(path)
#
#             for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
#
#                 left=box[0]
#                 top=box[1]
#                 right=box[2]+box[0]
#                 bottom=box[3]+box[1]
#                 # print(type(pic))
#                 image1=Image.fromarray(pic)
#                 # print(type(image1))
#
#                 crop_image = image1.crop((left, top, right, bottom))
#                 crop_image.save(path+'/'+str(numImg)+'_'+classNames[classId-1]+'.png')
#                 np_img=np.array(crop_image)
#                 list2.append(np_img)
#
#                 numImg=numImg+1
#                 # pre_img[-1]=(np_img,True)
#
#             for pic in list2:
#                 cv2.imshow('crop',pic)
#                 cv2.waitKey(500)
#                 cv2.destroyAllWindows()
#         else:
#             list2.append('yolo didnt found objects in this image')
#         list1.append(list2)
#         print(numImg)
#         dir=dir+1
#
#     return list1
#
# img = cv2.imread("images/lisbon.jpg")
# img1 = cv2.imread("images/V_DRONE_0011_007.png")
# img2 = cv2.imread("images/V_DRONE_09110_086.png")
# img3 = cv2.imread("images/V_DRONE_11030_325.png")
# img4 = cv2.imread("images/V_HELICOPTER_0011_012.png")
#
# list_pic=[]
#
# list_pic.append(img)
# list_pic.append(img1)
# list_pic.append(img2)
# list_pic.append(img3)
# list_pic.append(img4)
#
# pre_img=yolo_detect_return_dirs(list_pic)

# def fun(list):
#     list_for_sent=[]
#     if len(list)!=0:
#         for path in list:
#             print(path)
#             img = cv2.imread(path)
#             print(img)
#             list_for_sent.append(img)
#         pre_img = yolo_detect_return_dirs(list)
#         return pre_img
#     else:
#         return 'the list is empty'
# print('images/lisbon.jpg')
# list=['images/lisbon.jpg','images/V_DRONE_0011_007.png']
# fun(list)




# pic=cv2.imread("images/lisbon.jpg")
# cv2.imshow('',pic)
# cv2.waitKey(500)
# cv2.destroyAllWindows()

# left = box[0]
# bottom = box[1]
# right = box[2] + box[0]
# top = box[3] + box[1]
def yolo_detect_return_places_list(frame):
    classFile='yolo_file/coco.names'

    with open(classFile,'rt') as f:
        classNames=f.read().rstrip('\n').split('\n')
    configPath='yolo_file/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath='yolo_file/frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath,configPath)

    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(frame)

    types=[]
    places = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            myPlace=[box[0],box[1],box[2],box[3]]
            places.append(myPlace)
            types.append(classNames[classId-1])
    return places,types






