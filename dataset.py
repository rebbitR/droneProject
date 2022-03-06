
# create_dataset_from_csv:----------------------------
import cv2
import pandas as pd
from os import makedirs
from os.path import splitext, dirname, basename, join
import numpy
from PIL import Image
import numpy as np
from numpy import asarray
# def changeResolution(path,i):
#     # Open the image by specifying the image path.
#     image_file = Image.open(path)
#     image_file.save("changeReolotiution/frame"+i+".jpg", quality=32)

# Importing necessary functions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
from PIL import Image

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
        # cv2.imshow("cropped", img)
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
    return images_augmentation


path = 'tryyyyyyyyyyyy.jpg'
path2 = 'image_augmantion'


image_augmentation(path, 1, path2)
def changeResolution1(pathOfImage,numImage):
    img = Image.open(pathOfImage)
    resized_img = img.resize((224, 224))
    # resized_img.save("fixResolution/resized_image"+str(numImage)+".jpg")
    # path="fixResolution/resized_image"+str(numImage)+".jpg"
    return resized_img

#changeResolution("tryReolotiution/images.jpg")


def changeResolution2(frame):
    # numpydata = asarray(frame)
    # resized_img = numpydata.resize((32, 32))
    img = Image.fromarray(frame)
    resized_img = img.resize((224, 224))
    #resized_img.save("resized_image"+str(1)+".jpg")
    open_cv_image = numpy.array(resized_img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def white(img):
    # read image
    ht, wd, cc = img.shape
    # create new image of desired size and color (white) for padding
    ww = 224
    hh = 224
    color = (255, 255, 255)
    result = numpy.full((hh, ww, cc), color, dtype=numpy.uint8)
    # set offsets for top left corner
    xx = 0
    yy = 0
    # copy img image into center of result image
    result[yy:yy + ht, xx:xx + wd] = img
    return result

# function that get image and place and return fix image for the model
# the place mast be: [x,y,w,h]
def cut_place(image,place):

    placeInt=place[1:-2]
    placeInt=placeInt.split(',')
    x=round(float(placeInt[0]))
    y=round(float(placeInt[1]))
    w = round(float(placeInt[2]))
    h = round(float(placeInt[3]))
    count=h*w
    print(count, "pixels")
    buf_img=[]
    if(count>50,176):
        crop_img = image[y:y + h, x:x + w]
        images_augmentation=image_augmentation(crop_img,"image_augmentation")
        for img in images_augmentation:
            img=changeResolution2(img)
            buf_img.append(img)
    if(count<50,176):
        if(h<224 and w<224):
            crop_img = image[y:y + h, x:x + w]
            images_augmentation = image_augmentation(crop_img, "image_augmentation")
            for img in images_augmentation:
                img = white(crop_img,)
                buf_img.append(img)
        else:
            if(h<224):
                h=224
            if(w<224):
                w=224
            crop_img = image[y:y + h, x:x + w]
            images_augmentation = image_augmentation(crop_img, "image_augmentation")
            for img in images_augmentation:
                img = changeResolution2(img)
                buf_img.append(img)
    else:
        crop_img = image[y:y + h, x:x + w]
        images_augmentation = image_augmentation(crop_img, "image_augmentation")
        buf=images_augmentation

    # height,width,c=crop_img.shape
    # print(height*width, "pixels")
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(500)
    # cv2.destroyAllWindows()
    return buf

# function that get csv with path, and place for every image, and directory for the fix images
def create_dataset_from_csv(csv, frame_dir: str,name="image", ext="jpg"):
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
    for i in range(0,df.shape[0],5):
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
                b='['+newPlace[1]
                imgs1 = cut_place(image, a)
                img=[]
                for img1 in imgs1:
                    img.append(img1)
                imgs2 = cut_place(image, b)
                for img2 in imgs2:
                    img.append(img2)
            else:
                img=[]
                imgs=cut_place(image, place)
                for img in imgs:
                    img.append(img)
            #copy the crope image to the directory
            if numOfImages == 1:  #Save 0 second frame
                for i in img:
                    cv2.imwrite("{}_{}.{}".format(base_path, "0000", ext), i)
                    numOfImages += 1
            else:
                for i in img:
                    filled_numOfImages = str(numOfImages).zfill(4)
                    cv2.imwrite("{}_{}.{}".format(base_path, filled_numOfImages, ext),i)
                    print("{}_{}.{}".format(base_path, filled_numOfImages, ext))
                    numOfImages += 1
    print(numOfImages)

# # create the dataset
# path=['airplain_close_csv','airplain_medium_csv',
#       'bird_close_csv','bird_medium_csv',
#       'drone_close_csv','drone_distance_csv','drone_medium1_csv','drone_medium2_csv',
#       'helicopter_close_csv','helicopter_medium_csv']
# for i in path:
#     csvPath=i+'.csv'
#     name_dir='./create_dataset'
#     create_dataset_from_csv(csvPath,name_dir)


# split_train_test_validation:---------------------------------------------
from sklearn.model_selection import train_test_split
import os
import shutil

def split_train_test_validation(base_dir_airplain, base_dir_helicopter,base_dir_bird,base_dir_drone):

    airplain = []
    helicopter = []
    bird = []
    drone = []
    for i in sorted(os.listdir(base_dir_airplain)):  # go through the whole list of files (in yes)
        airplain.append(i)  # add the names of all the files to the array

    for i in sorted(os.listdir(base_dir_helicopter)):  # go through the whole list of files (in no)
        helicopter.append(i)  # add the names of all the files to the array

    for i in sorted(os.listdir(base_dir_bird)):  # go through the whole list of files (in no)
        bird.append(i)  # add the names of all the files to the array

    for i in sorted(os.listdir(base_dir_drone)):  # go through the whole list of files (in no)
        drone.append(i)  # add the names of all the files to the array

    # divide randomly into folders (for both classes):
    # train: 70 %
    # test: 20 %
    # validate: 10 %
    airplain_train_validate, airplain_test = train_test_split(airplain, test_size=0.2, random_state=1)  # puts 20% to yes_test
    airplain_train, airplain_validate = train_test_split(airplain_train_validate, test_size=0.12,
                                               random_state=1)  # puts 12% of yes_train_validate into yes_validate
    helicopter_train_validate, helicopter_test = train_test_split(helicopter, test_size=0.2, random_state=1)  # puts 20% to no_test
    helicopter_train, helicopter_validate = train_test_split(helicopter_train_validate, test_size=0.12,
                                             random_state=1)  # puts 12% of no_train_validate into no_validate
    bird_train_validate, bird_test = train_test_split(bird, test_size=0.2, random_state=1)  # puts 20% to no_test
    bird_train, bird_validate = train_test_split(bird_train_validate, test_size=0.12,
                                             random_state=1)  # puts 12% of no_train_validate into no_validate
    drone_train_validate, drone_test = train_test_split(drone, test_size=0.2, random_state=1)  # puts 20% to no_test
    drone_train, drone_validate = train_test_split(drone_train_validate, test_size=0.12,
                                             random_state=1)  # puts 12% of no_train_validate into no_validate


    dest = ''
    ori = ''

    # airplain
    for item in airplain_train:
        ori = base_dir_airplain + '/' + item  # original folder
        dest = os.path.join("Data/dataset/train/airplain/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in airplain_test:
        ori = base_dir_airplain + '/' + item  # original folder
        dest = os.path.join("Data/dataset/test/airplain/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in airplain_validate:
        ori = base_dir_airplain + '/' + item  # original folder
        dest = os.path.join("Data/dataset/validate/airplain/", item)  # destination folder
        shutil.copy(ori, dest)

    # helicopter
    for item in helicopter_train:
        ori = base_dir_helicopter + '/' + item  # original folder
        dest = os.path.join("Data/dataset/train/helicopter/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in helicopter_test:
        ori = base_dir_helicopter + '/' + item  # original folder
        dest = os.path.join("Data/dataset/test/helicopter/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in helicopter_validate:
        ori = base_dir_helicopter + '/' + item  # original folder
        dest = os.path.join("Data/dataset/validate/helicopter/", item)  # destination folder
        shutil.copy(ori, dest)

    # bird
    for item in bird_train:
        ori = base_dir_bird + '/' + item  # original folder
        dest = os.path.join("Data/dataset/train/bird/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in bird_test:
        ori = base_dir_bird + '/' + item  # original folder
        dest = os.path.join("Data/dataset/test/bird/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in bird_validate:
        ori = base_dir_bird + '/' + item  # original folder
        dest = os.path.join("Data/dataset/validate/bird/", item)  # destination folder
        shutil.copy(ori, dest)

    # drone
    for item in drone_train:
        ori = base_dir_drone + '/' + item  # original folder
        dest = os.path.join("Data/dataset/train/drone/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in drone_test:
        ori = base_dir_drone + '/' + item  # original folder
        dest = os.path.join("Data/dataset/test/drone/", item)  # destination folder
        shutil.copy(ori, dest)

    for item in drone_validate:
        ori = base_dir_drone + '/' + item  # original folder
        dest = os.path.join("Data/dataset/validate/drone/", item)  # destination folder
        shutil.copy(ori, dest)


#split_train_test_validation("Data/airplain", "Data/helicopter","Data/bird","Data/drone")