import cv2
from yolo import yolo_detect
from dataset import white,change_resolution
from model_functions import load_my_model
from log import write_to_log


class Frame:
    global size
    size=81
    global models_dict
    models_dict={'category_vgg16':'models_files/pred_drone_5_classes_restnet_50_3.h5',
                 'binary_vgg16':'models_files/model_vgg_s81.h5',
                 'resnet_50':'models_files/pred_drone_5_classes_restnet_50_4.h5'}

    def __init__(self,frame,id):
        self.frame=frame
        self.id=id
        self.objects=[]

    def show_img(self,img=[],txt='',waitKey=500):
        if (img==[]):
            img=self.frame
        cv2.imshow(txt, img)
        cv2.waitKey(waitKey)
        cv2.destroyAllWindows()

    def change_resolution(self,img):
        return change_resolution(img,size)

    def white(self,img):
        return white(img,size)

    def add_txt(self):
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale=1.3
        color=(255, 255, 255)
        thickness=1
        for my_obj in self.objects:
            cv2.putText(self.frame, my_obj.kindC, (my_obj.place.x, my_obj.place.y), font,
                            fontScale, color, thickness, cv2.LINE_AA)

    def cut_objects(self):
        mone=0
        for obj1 in self.objects:
            x = round(float(obj1.place.x))
            y = round(float(obj1.place.y))
            w = round(float(obj1.place.w))
            h = round(float(obj1.place.h))
            if (w>size or h>size):
                if w<size:
                    w=size
                elif h<size:
                    h=size
                crop_img = self.frame[y:y + h, x:x + w]
                crop_img = self.change_resolution(crop_img)
            else:
                w = size
                h = size
                crop_img = self.frame[y:y + h, x:x + w]
            self.objects[mone].objectC=crop_img
            mone=mone+1

    def detect_objects(self):
        places,types,confs = yolo_detect(self.frame)
        mone=0
        if len(places)!=0:
            for i in range(len(places)):
                object = Obj(i,places[i],yolo=types[i])
                self.objects.append(object)
                mone=mone+1

    def model_detect(self,model_type):
        classes = ['airplane', 'bird', 'drone', 'helicopter', 'other']
        if model_type == 'binary_vgg16':
            classes = ['yes', 'no']
        path_model=models_dict[model_type]
        for obg in self.objects:
            obg.object_classification(model_type,path_model,size,classes)

    def print_results_frame(self):
        write_to_log('--frame id: '+str(self.id))
        print('--frame id: '+str(self.id))
        write_to_log('  num objects: '+str(len(self.objects)))
        numObj = 0
        for my_obj in self.objects:
            write_to_log('   object number '+str(numObj))
            my_obj.print_results_obj()
            numObj = numObj + 1

    def add_rectangle(self, left, top, right, bottom, R, B, G):
        return cv2.rectangle(self.frame, (left, top), (right, bottom), (B, G, R), 2)

    def result(self):
        if len(self.objects)!=0:
            for myObg in self.objects:
                if myObg.kindC=='drone':
                    # red
                    R=255;B=0;G=0
                elif myObg.kindC=='airplane':
                    # blue
                    R=0;B=255;G=0
                elif myObg.kindC=='bird':
                    # green
                    R=0;B=0;G=255
                elif myObg.kindC=='helicopter':
                    # black
                    R=0;B=0;G=0
                else:
                    # white
                    R=255;B=255;G=255
                self.frame=self.add_rectangle(
                                      myObg.place.x,
                                      myObg.place.y+myObg.place.h,
                                      myObg.place.x+myObg.place.w,
                                      myObg.place.y,
                                      R,B,G)



class Place:
    def __init__(self,x,y,w,h):
        self.x=x
        self.y=y
        self.w=w
        self.h=h



class Obj:
    def __init__(self,id,place1,object=None,kind=None,yolo='None',category_vgg16='None',binary_vgg16='None',resnet_50='None'):
        my_place=Place(place1[0],place1[1],place1[2],place1[3])
        self.place=my_place
        self.objectC=object
        self.models={'yolo':yolo,'category_vgg16':category_vgg16,'binary_vgg16':binary_vgg16,'resnet_50':resnet_50}
        self.kindC=kind
        self.id=id


    def object_classification(self,model_type,path_model,size,classes):
        path_img = "object" + ".png"
        cv2.imwrite(path_img, self.objectC)
        output, i, kind=load_my_model(path_model,classes,path_img,size)
        self.kindC=kind
        self.models[model_type]=kind

    def print_results_obj(self):
        write_to_log(self.models)

