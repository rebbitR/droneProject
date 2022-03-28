from bs4 import BeautifulSoup
import cv2
import glob, os


save_path = 'datast_xml_result'

name_of_file =1

completeName = os.path.join(save_path, name_of_file+".txt")
os.


def cut_place(image,place,base_path,name):

    x=round(place[0])
    y=round(place[1])
    w=round(place[2])
    h=round(place[3])
    crop_image=image[y:y + h, x:x + w]
    cv2.imwrite("crop_{}.{}".format(base_path, name, 'jpg'), crop_image)

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
    return place

def fun(path,path_dir_save):
    os.chdir(path)
    for file in glob.glob("*.xml"):
        place=read_from_xml(file)
        img=cv2.imread(file[0:len(file)-4]+'.jpg')
        cut_place(img,place,path_dir_save,file[0:len(file)-4])

fun('dataset_xml_format','datast_xml_result')