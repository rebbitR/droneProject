import cv2
import numpy as np
from PIL import Image
from load_model import model
from yolo import yolo_detect_return_places_list


def rectangle(frame,left,top,right,bottom,R,B,G):
    return cv2.rectangle(frame,(left,top),(right,bottom),(B,G,R),2)
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
        print(type(img))
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

    # # function that get image and place and return fix image for the model:
    # # the place must be: [x,y,w,h]
    # def cut_objects(self):
    #
    #     mone=0
    #     for obj1 in self.objectsC:
    #         placeInt = obj1.placeC[1:-2]
    #         placeInt = placeInt.split(',')
    #         x = round(float(placeInt[0]))
    #         y = round(float(placeInt[1]))
    #         w = round(float(placeInt[2]))
    #         h = round(float(placeInt[3]))
    #         count = h * w
    #         print(count, "pixels")
    #         if (count > 50176):
    #             crop_img = self.frameC[y:y + h, x:x + w]
    #             crop_img = frame.changeResolution2(crop_img)
    #         if (count < 50176):
    #             if (h < 224 and w < 224):
    #                 crop_img = self.frameC[y:y + h, x:x + w]
    #                 crop_img = frame.white(crop_img)
    #             else:
    #                 if (h < 224):
    #                     h = 224
    #                 if (w < 224):
    #                     w = 224
    #                 crop_img = self.frameC[y:y + h, x:x + w]
    #                 crop_img = frame.changeResolution2(crop_img)
    #         else:
    #             crop_img = self.frameC[y:y + h, x:x + w]
    #
    #         self.objectsC[mone].objectC=crop_img
    #         mone=mone+1
    #
    #         height, width, c = crop_img.shape
    #         print(height * width, "pixels")
    #         # cv2.imshow("cropped", crop_img)
    #         # cv2.waitKey(500)
    #         # cv2.destroyAllWindows()

    # function that get image and place and return fix image for the model:
    # the place must be: [x,y,w,h]
    def cut_objects(self):

        mone=0
        for obj1 in self.objectsC:

            x = round(float(obj1.placeC.xC))
            y = round(float(obj1.placeC.yC))
            w = round(float(obj1.placeC.wC))
            h = round(float(obj1.placeC.hC))


            # image1 = Image.fromarray(self.frameC)
            # crop_image = image1.crop((x, y, h, w))
            # np_img = np.array(crop_image)
            crop_img = self.frameC[obj1.placeC.yC:obj1.placeC.yC + obj1.placeC.hC, obj1.placeC.xC:obj1.placeC.xC + obj1.placeC.wC]
            cv2.imshow('before', crop_img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            crop_img = self.white(crop_img)
            # crop_img = self.changeResolution2(np_img)
            cv2.imshow('after', crop_img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

            self.objectsC[mone].objectC=crop_img
            mone=mone+1


    def yolo_detect(self):
        places = yolo_detect_return_places_list(self.frameC)
        for myPlace in places:
            object = obj(myPlace)
            self.objectsC.append(object)





    def model(self):
        mone=0
        for obj1 in self.objectsC:
            print(type(obj1))
            cv2.imwrite("object.png", obj1.objectC)
            i=model("object.png")
            classes = ['airplain', 'bird', 'drone', 'helicopter']
            print(classes[i])
            self.objectsC[mone].kindC = classes[i]
            mone = mone + 1



    # פונקציה המסמנת על התמונה את האוביקטים שנמצאו וסוגם
    def result(self):
        if len(self.objectsC)!=0:
            for myObg in self.objectsC:
                if myObg.kindC=='drone':
                    # red
                    R=255
                    B=0
                    G=0
                elif myObg.kindC=='airplain':
                    # blue
                    R=0
                    B=255
                    G=0
                elif myObg.kindC=='bird':
                    # green
                    R=0
                    B=0
                    G=255
                elif myObg.kindC=='helicopter':
                    # black
                    R=0
                    B=0
                    G=0
                self.frameC=rectangle(self.frameC,
                                      myObg.placeC.xC,
                                      myObg.placeC.yC+myObg.placeC.hC,
                                      myObg.placeC.xC+myObg.placeC.wC,
                                      myObg.placeC.yC,
                                      R,B,G)



class place:
    def __init__(self,x,y,w,h):
        print(y)
        self.xC=x
        self.yC=y
        self.wC=w
        self.hC=h



class obj:
    def __init__(self,place1,object=None,kind=None):
        my_place=place(place1[0],place1[1],place1[2],place1[3])
        self.placeC=my_place
        self.objectC=object
        self.kindC=kind

# class try1:
#     def __init__(self,x):
#         self.x=x
#     def replace(self):
#         self.x=100
#     def ret(self):
#         return self.x