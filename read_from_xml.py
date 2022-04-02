from bs4 import BeautifulSoup
import cv2
import os

def cut_place_from_path(path,x,y,h,w):
    img=cv2.imread(path)
    # cv2.imshow('', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    crop_image=img[y:y + h, x:x + w]
    cv2.imshow('', crop_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cut_place(image,place,path_dir_save,name):

    x=round(place[0])
    y=round(place[1])
    w=round(place[2])
    h=round(place[3])
    crop_image=image[y:y + h, x:x + w]
    status=cv2.imwrite(path_dir_save+'/'+name+'.jpg',crop_image)
    print(status)



def read_from_xml(path_xml):

    place=[]
    with open(path_xml, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")
    xmin = Bs_data.find_all('xmin')
    ymin = Bs_data.find_all('ymin')
    xmax = Bs_data.find_all('xmax')
    ymax = Bs_data.find_all('ymax')
    x=float(xmin.pop().text)
    place.append(x)
    y=float(ymin.pop().text)
    place.append(y)
    w2=float(xmax.pop().text)
    place.append(w2-x)
    h2=float(ymax.pop().text)
    place.append(h2-y)
    print(place)
    return place

def create_dataset_from_xml(path,path_dir_save):

    for filename in os.listdir(path):
        print(filename[0:len(filename)-4]+'.xml')
        place=read_from_xml(path+'/'+filename[0:len(filename)-4]+'.xml')
        img=cv2.imread(path+'/'+filename[0:len(filename)-4]+'.jpg')
        cut_place(img,place,path_dir_save,filename[0:len(filename)-4])

# fun('dataset_xml_format','crop_by_xml_result')
# cut_place_from_path('0006.jpg',210,510,180,630)

