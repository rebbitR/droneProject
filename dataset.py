
# create_dataset_from_csv:----------------------------
from bs4 import BeautifulSoup
import pandas as pd
from os import makedirs
from os.path import splitext, dirname, basename, join
import numpy
from yolo import yolo_detect_return_places_list
import numpy as np
from numpy import asarray
from PIL import Image


# Importing necessary functions
from keras.preprocessing.image import ImageDataGenerator,img_to_array, array_to_img, load_img
import os
import cv2

# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
def image_augmentation(img, p):
    datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.5, 1.5))

    # Converting the input sample image to an array
    x = img_to_array(img)
    # Reshaping the input image
    x = x.reshape((1,) + x.shape)

    # Generating and saving 5 augmented samples
    # using the above defined parameters.
    i = 300
    for batch in datagen.flow(x, batch_size=1, save_to_dir=p, save_prefix=1, save_format='png'):

        i += 1
        if i > 305:
            break

    images_augmentation=[]
    for path in os.listdir(p):
        img_path=p+'/'+path
        img = cv2.imread(img_path)
        images_augmentation.append(img)
        os.remove(img_path)
        #cv2.imshow("cropped_image_agm", img)
        #cv2.waitKey(500)
        #cv2.destroyAllWindows()
    return images_augmentation


def changeResolution1(pathOfImage,numImage):
    img = Image.open(pathOfImage)
    resized_img = img.resize((224, 224))
    # resized_img.save("fixResolution/resized_image"+str(numImage)+".jpg")
    # path="fixResolution/resized_image"+str(numImage)+".jpg"
    return resized_img

#changeResolution("tryReolotiution/images.jpg")


def changeResolution2(frame,size):
    # numpydata = asarray(frame)
    # resized_img = numpydata.resize((size, size))
    img = Image.fromarray(frame)
    resized_img = img.resize((size, size))
    #resized_img.save("resized_image"+str(1)+".jpg")
    open_cv_image = numpy.array(resized_img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


# function that return image whith white bakeground for size 224/224 to sent the model
def white(img,size):
    # read image
    ht, wd, cc = img.shape
    # create new image of desired size and color (white) for padding
    ww = size
    hh = size
    color = (255, 255, 255)
    result = numpy.full((hh, ww, cc), color, dtype=numpy.uint8)
    # set offsets for top left corner
    xx = 0
    yy = 0
    # copy img image into center of result image
    result[yy:yy + ht, xx:xx + wd] = img
    return result

# # function that get image and place and return fix image for the model
# # the place mast be: [x,y,w,h]
# def cut_place(image,place):
#
#     placeInt=place[1:-2]
#     placeInt=placeInt.split(',')
#     x=round(float(placeInt[0]))
#     y=round(float(placeInt[1]))
#     w = round(float(placeInt[2]))
#     h = round(float(placeInt[3]))
#     count=h*w
#     print(count, "pixels")
#     buf_img=[]
#     if(count>50,176):
#         crop_img = image[y:y + h, x:x + w]
#         images_augmentation=image_augmentation(crop_img,"image_augmentation")
#         for img in images_augmentation:
#             img=changeResolution2(img)
#             buf_img.append(img)
#     if(count<50,176):
#         if(h<224 and w<224):
#             crop_img = image[y:y + h, x:x + w]
#             images_augmentation = image_augmentation(crop_img, "image_augmentation")
#             for img in images_augmentation:
#                 img = white(crop_img,)
#                 buf_img.append(img)
#         else:
#             if(h<224):
#                 h=224
#             if(w<224):
#                 w=224
#             crop_img = image[y:y + h, x:x + w]
#             images_augmentation = image_augmentation(crop_img, "image_augmentation")
#             for img in images_augmentation:
#                 img = changeResolution2(img)
#                 buf_img.append(img)
#     else:
#         crop_img = image[y:y + h, x:x + w]
#         images_augmentation = image_augmentation(crop_img, "image_augmentation")
#         buf=images_augmentation
#
#     # height,width,c=crop_img.shape
#     # print(height*width, "pixels")
#     # cv2.imshow("cropped", crop_img)
#     # cv2.waitKey(500)
#     # cv2.destroyAllWindows()
#     return buf

# function that get image and place_arr and return fix image for the model
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
        images_augmentation=image_augmentation(crop_img,"image_augmentation")
        for img in images_augmentation:
            # print("1")
            img=changeResolution2(img,size)
            buf_img.append(img)
    else:
        w=size
        h=size
        crop_img = image[y:y + h, x:x + w]
        buf_img=image_augmentation(crop_img,"image_augmentation")

    # for img in buf_img:
    #     cv2.imshow("cropped", img)
    #     cv2.waitKey(10)
    #     cv2.destroyAllWindows()
    return buf_img

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
    # print(place)
    return place



def create_dataset_from_xml1(path,path_dir_save,size):

    for filename in os.listdir(path):
        # print(filename[0:len(filename)-4]+'.xml')
        place=read_from_xml(path+'/'+filename[0:len(filename)-4]+'.xml')
        img=cv2.imread(path+'/'+filename[0:len(filename)-4]+'.jpg')
        buf=cut_place(img,place,size)
        for i in range(len(buf)):
            cv2.imwrite(path_dir_save + '/' + filename[0:len(filename)-4] + '_'+str(i)+'.jpg', buf[i])


def create_dataset_with_yolo_places(original_path,save_in,size):
    for filename in os.listdir(original_path):
        path=original_path+'/'+filename
        print(path)
        img=cv2.imread(path)
        places,types=yolo_detect_return_places_list(img)
        print(places)
        if len(places)!=0:
            for p in range(len(places)):
                buf = cut_place(img, places[p], size)
                for i in range(len(buf)):
                    cv2.imwrite(save_in + '/' + filename[0:len(filename) - 4] + '_' + str(p)+ '_' + str(i) + '.jpg', buf[i])

# create_dataset_with_yolo_places('try_original_img','try',81)

def fix_place_to_arr(place):
    placeInt=place[1:-2]
    placeInt=placeInt.split(',')
    return placeInt

# function that get csv whith path, and place for every image, and directory for the fix images
def create_dataset_from_csv(csv, frame_dir: str,size,name="image", ext="jpg"):
    #create dir for the crope images in this csv
    v_name = splitext(basename(csv))[0]
    video_path_arr = csv.split('/')
    print(video_path_arr)
    if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
        frame_dir = dirname(frame_dir)
    frame_dir_ = join(frame_dir, v_name)

    makedirs(frame_dir_, exist_ok=True)
    base_path = join(frame_dir_, name)
    print(base_path)

    df = pd.read_csv(csv)
    numOfImages = 1
    for i in range(0,df.shape[0],17):
        filePath = df['imageFilename'][i]
        filePath = filePath.replace('\\', '/')
        filePath = filePath.replace('\'', '')
        print(filePath)
        place=''
        if df['BIRD'][i]!='[]':
            place = df['BIRD'][i]
        if df['AIRPLANE'][i] != '[]':
            place = df['AIRPLANE'][i]
        if df['DRONE'][i]!='[]':
            place = df['DRONE'][i]
        if df['HELICOPTER'][i]!='[]':
            place = df['HELICOPTER'][i]
        if place[0]=='[':
            print(place)
            image = cv2.imread(filePath, 1)
            if place.find(';')!=-1:
                newPlace=place.split(';')
                a=newPlace[0]+']'
                a=fix_place_to_arr(a)
                b='['+newPlace[1]
                b=fix_place_to_arr(b)
                img1 = cut_place(image, a,size)
                img=[]
                img.append(img1)
                img2 = cut_place(image, b,size)
                img.append(img2)
            else:
                img=[]
                place=fix_place_to_arr(place)
                img1=cut_place(image, place,size)
                img.append(img1)

            #copy the crope image to the directory
            if numOfImages == 1:  #Save 0 second frame
                for i in img:
                    for j in i:
                        cv2.imwrite("{}_{}.{}".format(base_path, "0000", ext), j)
                        numOfImages += 1
            else:
                for i in img:
                    for j in i:
                        filled_numOfImages = str(numOfImages).zfill(4)
                        cv2.imwrite("{}_{}.{}".format(base_path, filled_numOfImages, ext),j)
                        print("{}_{}.{}".format(base_path, filled_numOfImages, ext))
                        numOfImages += 1
    print(numOfImages)

# # create the dataset/// for model number 1:
# path=['airplain_close_csv','airplain_medium_csv',
#       'bird_close_csv','bird_medium_csv',
#       'drone_close_csv','drone_distance_csv','drone_medium1_csv','drone_medium2_csv',
#       'helicopter_close_csv','helicopter_medium_csv']
# for i in path:
#     csvPath=i+'.csv'
#     name_dir='./create_dataset'
#     create_dataset_from_csv(csvPath,name_dir)

# # create the dataset/// for model number 2:
# size=81
# path=['airplane_close_csv','airplane_medium_csv',
#       'bird_close_csv','bird_medium_csv',
#       'drone_close_csv','drone_distance_csv','drone_medium1_csv','drone_medium2_csv',
#       'helicopter_close_csv','helicopter_medium_csv']
# for i in path:
#     csvPath=i+'.csv'
#     name_dir='./create_dataset_s81'
#     create_dataset_from_csv(csvPath,name_dir,size)
# create_dataset_from_xml1('dataset_xml_format','drone_from_xml',size)
# create_dataset_with_yolo_places('Hot Air Balloon','hot_hair_balloon_dataset',size)
# create_dataset_with_yolo_places('kites','kite_dataset',size)
# create_dataset_with_yolo_places('baloon','baloon_dataset',size)

# split_train_test_validation:---------------------------------------------
# from sklearn.model_selection import train_test_split
import os
import shutil

def split_train_test_validation(base_dir_airplane, base_dir_helicopter,base_dir_bird,base_dir_drone,base_dir_other):

    airplane = []
    helicopter = []
    bird = []
    drone = []
    other=[]
    for i in sorted(os.listdir(base_dir_airplane)):  # go through the whole list of files (in airplane)
        airplane.append(i)  # add the names of all the files to the array

    for i in sorted(os.listdir(base_dir_helicopter)):  # go through the whole list of files (in helicopter)
        helicopter.append(i)  # add the names of all the files to the array

    for i in sorted(os.listdir(base_dir_bird)):  # go through the whole list of files (in bird)
        bird.append(i)  # add the names of all the files to the array

    for i in sorted(os.listdir(base_dir_drone)):  # go through the whole list of files (in drone)
        drone.append(i)  # add the names of all the files to the array

    for i in sorted(os.listdir(base_dir_other)):  # go through the whole list of files (in other)
        other.append(i)  # add the names of all the files to the array

    # divide randomly into folders (for both classes):
    # train: 70 %
    # test: 20 %
    # validate: 10 %
    airplane_train_validate, airplane_test = train_test_split(airplane, test_size=0.2, random_state=1)  # puts 20% to airplane_test
    airplane_train, airplane_validate = train_test_split(airplane_train_validate, test_size=0.12,
                                               random_state=1)  # puts 12% of airplane_train_validate into airplane_validate
    helicopter_train_validate, helicopter_test = train_test_split(helicopter, test_size=0.2, random_state=1)  # puts 20% to helicopter_test
    helicopter_train, helicopter_validate = train_test_split(helicopter_train_validate, test_size=0.12,
                                             random_state=1)  # puts 12% of helicopter_train_validate into helicopter_validate
    bird_train_validate, bird_test = train_test_split(bird, test_size=0.2, random_state=1)  # puts 20% to bird_test
    bird_train, bird_validate = train_test_split(bird_train_validate, test_size=0.12,
                                             random_state=1)  # puts 12% of bird_train_validate into bird_validate
    drone_train_validate, drone_test = train_test_split(drone, test_size=0.2, random_state=1)  # puts 20% to drone_test
    drone_train, drone_validate = train_test_split(drone_train_validate, test_size=0.12,
                                             random_state=1)  # puts 12% of drone_train_validate into drone_validate
    other_train_validate, other_test = train_test_split(other, test_size=0.2, random_state=1)  # puts 20% to other_test
    other_train, other_validate = train_test_split(other_train_validate, test_size=0.12,
                                             random_state=1)  # puts 12% of other_train_validate into other_validate


    dest = ''
    ori = ''

    # airplane
    for item in airplane_train:
        ori = base_dir_airplane + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/train/airplane/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in airplane_test:
        ori = base_dir_airplane + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/test/airplane/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in airplane_validate:
        ori = base_dir_airplane + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/validate/airplane/", item)  # destination folder
        shutil.copy(ori, dest)

    # helicopter
    for item in helicopter_train:
        ori = base_dir_helicopter + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/train/helicopter/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in helicopter_test:
        ori = base_dir_helicopter + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/test/helicopter/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in helicopter_validate:
        ori = base_dir_helicopter + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/validate/helicopter/", item)  # destination folder
        shutil.copy(ori, dest)

    # bird
    for item in bird_train:
        ori = base_dir_bird + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/train/bird/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in bird_test:
        ori = base_dir_bird + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/test/bird/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in bird_validate:
        ori = base_dir_bird + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/validate/bird/", item)  # destination folder
        shutil.copy(ori, dest)

    # drone
    for item in drone_train:
        ori = base_dir_drone + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/train/drone/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in drone_test:
        ori = base_dir_drone + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/test/drone/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in drone_validate:
        ori = base_dir_drone + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/validate/drone/", item)  # destination folder
        shutil.copy(ori, dest)

    # other
    for item in other_train:
        ori = base_dir_other + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/train/other/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in other_test:
        ori = base_dir_other + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/test/other/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in other_validate:
        ori = base_dir_other + '/' + item  # original folder
        dest = os.path.join("dataset_2_classes/dataset/validate/other/", item)  # destination folder
        shutil.copy(ori, dest)


#split_train_test_validation("Data/airplain", "Data/helicopter","Data/bird","Data/drone")
# split_train_test_validation("dataset_2_classes/airplane", "dataset_2_classes/helicopter","dataset_2_classes/bird","dataset_2_classes/drone","dataset_2_classes/other")