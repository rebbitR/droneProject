import cv2
import numpy as np
from PIL import Image
from load_model import my_model
from yolo import yolo_detect_return_places_list



class frame:
    global size
    size=81
    def __init__(self,frame,second):
        self.frameC=frame
        self.secondC=second
        self.objectsC=[]

    # def changeResolution1(self,pathOfImage, numImage):
    #     img = Image.open(pathOfImage)
    #     resized_img = img.resize((size, size))
    #     # resized_img.save("fixResolution/resized_image"+str(numImage)+".jpg")
    #     # path="fixResolution/resized_image"+str(numImage)+".jpg"
    #     return resized_img

    def show_img(self,img=[],txt='',waitKey=500):
        if (img==[]):
            img=self.frameC
        cv2.imshow(txt, img)
        cv2.waitKey(waitKey)
        cv2.destroyAllWindows()

    def changeResolution2(self,frame):
        # numpydata = asarray(frame)
        # resized_img = numpydata.resize((size, size))
        img = Image.fromarray(frame)
        resized_img = img.resize((size, size))
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
        ww = size
        hh = size
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

            x = round(float(obj1.placeC.xC))
            y = round(float(obj1.placeC.yC))
            w = round(float(obj1.placeC.wC))
            h = round(float(obj1.placeC.hC))

            if (w>size or h>size):
                if w<size:
                    w=size
                elif h<size:
                    h=size
                crop_img = self.frameC[y:y + h, x:x + w]
                # self.show_img(img=crop_img,txt='before')
                crop_img = self.changeResolution2(crop_img)
            else:
                # # with white:
                # crop_img = self.frameC[y:y + h, x:x + w]
                # # self.show_img(img=crop_img,txt='before')
                # crop_img = self.white(crop_img)

                # without white:
                w = size
                h = size
                crop_img = self.frameC[y:y + h, x:x + w]
            # self.show_img(img=crop_img,txt='after')
            self.objectsC[mone].objectC=crop_img
            mone=mone+1


    def yolo_detect(self):
        places,types,confs = yolo_detect_return_places_list(self.frameC)
        mone=0
        # print('yolo:')
        if len(places)!=0:
            for i in range(len(places)):
                # print(str(mone)+' '+types[mone]+" place: x,y,w,h: "+str(places[i])+', left,top,right,bottom: '+str(places[i][0])+","+str(places[i][1]+places[i][3])+','+str(places[i][0]+places[i][2])+","+str(places[i][1]))
                object = obj(places[i],yolo=types[i])
                self.objectsC.append(object)
                mone=mone+1


    def model(self,type):
        mone=0
        # print('model:')
        for obj1 in self.objectsC:
            cv2.imwrite("object.png", obj1.objectC)
            i,kind = my_model("object.png", size, type)
            if kind=='ERROR':
                break
            print(str(mone)+' '+kind)
            self.objectsC[mone].models[type] = kind
            mone = mone + 1

    # def check(self):
    #     if len(self.objectsC)!=0:
    #         for my_obg in self.objectsC:
    #             if my_obg.models['yolo']=='airplane' or my_obg.models['yolo']=='bird' or my_obg.models['yolo']=='kite':
    #                 if my_obg.models['category']=='drone' or my_obg.models['binary']=='drone'
    #                     my_obg.kindC=='drone'
    #             else:



    def rectangle(self,frame, left, top, right, bottom, R, B, G):
        return cv2.rectangle(frame, (left, top), (right, bottom), (B, G, R), 2)

    # פונקציה המסמנת על התמונה את האוביקטים שנמצאו וסוגם
    def result(self):
        if len(self.objectsC)!=0:
            for myObg in self.objectsC:
                myObg.kindC=myObg.models['category']
                if myObg.kindC=='drone':
                    # red
                    R=255
                    B=0
                    G=0
                elif myObg.kindC=='airplane':
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
                else:
                    # white
                    R=255
                    B=255
                    G=255
                self.frameC=self.rectangle(self.frameC,
                                      myObg.placeC.xC,
                                      myObg.placeC.yC+myObg.placeC.hC,
                                      myObg.placeC.xC+myObg.placeC.wC,
                                      myObg.placeC.yC,
                                      R,B,G)



class place:
    def __init__(self,x,y,w,h):
        self.xC=x
        self.yC=y
        self.wC=w
        self.hC=h



class obj:
    def __init__(self,place1,object=None,kind=None,yolo='nuthing',category='nuthing',binary='nuthing'):
        my_place=place(place1[0],place1[1],place1[2],place1[3])
        self.placeC=my_place
        self.objectC=object
        self.models={'yolo':yolo,'category':category,'binary':binary}
        self.kindC=kind

