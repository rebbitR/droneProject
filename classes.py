import cv2
import numpy as np
from PIL import Image

from model import model

class frame:
    def __init__(self,frame,second):
        self.frameC=frame
        self.secondC=second
        self.objectsC=[]

    def changeResolution1(self,pathOfImage, numImage):
        img = Image.open(pathOfImage)
        resized_img = img.resize((224, 224))
        # resized_img.save("fixResolution/resized_image"+str(numImage)+".jpg")
        # path="fixResolution/resized_image"+str(numImage)+".jpg"
        return resized_img

    # changeResolution("tryReolotiution/images.jpg")

    def changeResolution2(self,frame):
        # numpydata = asarray(frame)
        # resized_img = numpydata.resize((32, 32))
        img = Image.fromarray(frame)
        resized_img = img.resize((224, 224))
        # resized_img.save("resized_image"+str(1)+".jpg")
        open_cv_image = np.array(resized_img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image

    # cut object by place and white background:---------------------------
    # function that return image with white background for size 224/224 to send the model
    def white(self,img):
        # read image
        ht, wd, cc = img.shape
        # create new image of desired size and color (white) for padding
        ww = 224
        hh = 224
        color = (255, 255, 255)
        result = np.full((hh, ww, cc), color, dtype=np.uint8)
        # set offsets for top left corner
        xx = 0
        yy = 0
        # copy img image into center of result image
        result[yy:yy + ht, xx:xx + wd] = img
        return result

    # function that get image and place and return fix image for the model:
    # the place must be: [x,y,w,h]
    def cut_objects(self):

        mone=0
        for obj1 in self.objectsC:
            placeInt = obj1.placeC[1:-2]
            placeInt = placeInt.split(',')
            x = round(float(placeInt[0]))
            y = round(float(placeInt[1]))
            w = round(float(placeInt[2]))
            h = round(float(placeInt[3]))
            count = h * w
            print(count, "pixels")
            if (count > 50176):
                crop_img = self.frameC[y:y + h, x:x + w]
                crop_img = frame.changeResolution2(crop_img)
            if (count < 50176):
                if (h < 224 and w < 224):
                    crop_img = self.frameC[y:y + h, x:x + w]
                    crop_img = frame.white(crop_img)
                else:
                    if (h < 224):
                        h = 224
                    if (w < 224):
                        w = 224
                    crop_img = self.frameC[y:y + h, x:x + w]
                    crop_img = frame.changeResolution2(crop_img)
            else:
                crop_img = self.frameC[y:y + h, x:x + w]

            self.objectsC[mone].objectC=crop_img
            mone=mone+1

            height, width, c = crop_img.shape
            print(height * width, "pixels")
            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey(500)
            # cv2.destroyAllWindows()




    def yolo_detect(self):

        # classFile = 'yolo_file/coco.names'
        # with open(classFile, 'rt') as f:
        #     classNames = f.read().rstrip('\n').split('\n')

        configPath = 'yolo_file/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'yolo_file/frozen_inference_graph.pb'
        net = cv2.dnn_DetectionModel(weightsPath, configPath)

        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        classIds, confs, bbox = net.detect(np.asarray(self.frameC))

        numImg = 0

        if len(classIds) != 0:


            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                left = box[0]
                top = box[1]
                right = box[2] + box[0]
                bottom = box[3] + box[1]

                myPlace=place(left,top,right,bottom)
                object = obj(myPlace)
                self.objectsC.append(object)

                numImg = numImg + 1

    def model(self):
        # self.objectsC[i].kindC=kind
        return 


    # def find_kinds_with_model(self):
    #     for object in self.objectsC:
    #         object.kindC=model(object.objectC)



    # פונקציה המסמנת על התמונה את האוביקטים שנמצאו וסוגם
    def result(self):
        if self.objectsC!=[]:
            for i in self.objectsC:
                if i.kindC=='Drone':
                    # 0=x, 1=y, 2=w, 3=h
                    # blue
                    self.frameC=cv2.rectangle(self.frameC,
                                  (i.placeC[0], i.placeC[1]),
                                  (i.placeC[0] + i.placeC[2],
                                   i.placeC[1] + i.placeC[3]),
                                  (255,0,0), 2)
                if i.kindC=='Airplain':
                    # 0=x, 1=y, 2=w, 3=h
                    # green
                    self.frameC=cv2.rectangle(self.frameC,
                                  (i.placeC[0], i.placeC[1]),
                                  (i.placeC[0] + i.placeC[2],
                                   i.placeC[1] + i.placeC[3]),
                                  (0, 255, 0), 2)
                if i.kindC=='Bird':
                    # 0=x, 1=y, 2=w, 3=h
                    # red
                    self.frameC=cv2.rectangle(self.frameC,
                                  (i.placeC[0], i.placeC[1]),
                                  (i.placeC[0] + i.placeC[2],
                                   i.placeC[1] + i.placeC[3]),
                                  (0, 0, 255), 2)
                if i.kindC=='Helicopter':
                    # 0=x, 1=y, 2=w, 3=h
                    # black
                    self.frameC=cv2.rectangle(self.frameC,
                                  (i.placeC[0], i.placeC[1]),
                                  (i.placeC[0] + i.placeC[2],
                                   i.placeC[1] + i.placeC[3]),
                                  (200, 200, 55), 2)




class obj:
    def __init__(self,place,object=None,kind=None):
        my_place=place(place[0],place[1],place[2],place[3])
        self.placeC=my_place
        self.objectC=object
        self.kindC=kind

class place:
    def __init__(self,x,y,h,w):
        self.xC=x
        self.yC=y
        self.hC=h
        self.wC=w