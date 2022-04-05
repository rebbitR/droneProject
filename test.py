import numpy
import pandas as pd
from os import makedirs
from os.path import splitext, dirname, basename, join
from resolution import changeResolution2

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
        #cv2.imshow("cropped_image_agm", img)
        #cv2.waitKey(500)
        #cv2.destroyAllWindows()
    return images_augmentation



# path = 'tryyyyyyyyyyyy.jpg'
# path2 = 'image_augmantion'
#
#
# image_augmentation(path, 1, path2)


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

# function that get image and place and return fix image for the model
# the place mast be: [x,y,w,h]
def cut_place(image,place,size):

    placeInt=place[1:-2]
    placeInt=placeInt.split(',')
    x=round(float(placeInt[0]))
    y=round(float(placeInt[1]))
    w = round(float(placeInt[2]))
    h = round(float(placeInt[3]))
    count=h*w
    print(count, "pixels")
    buf_img=[]
    if(h>size or w>size):
        if(h<size):
            h=size
        elif(w<size):
            w=size
        crop_img = image[y:y + h, x:x + w]
        images_augmentation=image_augmentation(crop_img,"image_augmentation")
        for img in images_augmentation:
            print("1")
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

# פונקציה שמקבלת מערך של תמונות עם מיקום של אוביקטים ומחזירה מערך עם האוביקטים לשליחה למודל
def buf_cut_places(buf):
    for image in buf:
        for object in image._objects:
            object._objectC=cut_place(image,object._placeC)
    return buf

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
                b='['+newPlace[1]
                img1 = cut_place(image, a,size)
                img=[]
                img.append(img1)
                img2 = cut_place(image, b,size)
                img.append(img2)
            else:
                img=[]
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



# create the dataset
path=['airplain_close_csv','airplain_medium_csv',
      'bird_close_csv','bird_medium_csv',
      'drone_close_csv','drone_distance_csv','drone_medium1_csv','drone_medium2_csv',
      'helicopter_close_csv','helicopter_medium_csv']
for i in path:
    csvPath=i+'.csv'
    name_dir='./create_datasetTry_81'
    size=81
    create_dataset_from_csv(csvPath,name_dir,size)



# filePath='BIRD_MEDIUM\V_BIRD_04621_311.png'
# image = cv2.imread(filePath, 1)
# place="[125,181,22,29]"
# detect_face(image,place)
# filePath='resized_image1.jpg'
# image = cv2.imread(filePath, 1)
# height, width, c = image.shape
# print(height * width, "pixels")

# # ----------------------------------------
# import cv2
# import numpy
# import pandas as pd
# from os import makedirs
# from os.path import splitext, dirname, basename, join
# from resolution import changeResolution2
#
# # Importing necessary functions
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# import os
# import cv2
# from PIL import Image
#
# # Initialising the ImageDataGenerator class.
# # We will pass in the augmentation parameters in the constructor.
# def image_augmentation(img, p):
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         brightness_range=(0.5, 1.5))
#
#     # Converting the input sample image to an array
#     x = img_to_array(img)
#     # Reshaping the input image
#     x = x.reshape((1,) + x.shape)
#
#     # Generating and saving 5 augmented samples
#     # using the above defined parameters.
#     i = 300
#     for batch in datagen.flow(x, batch_size=1, save_to_dir=p, save_prefix=1, save_format='png'):
#
#         i += 1
#         if i > 305:
#             break
#
#     images_augmentation=[]
#     for path in os.listdir(p):
#         img_path=p+'/'+path
#         img = cv2.imread(img_path)
#         images_augmentation.append(img)
#         os.remove(img_path)
#         # cv2.imshow("cropped", img)
#         # cv2.waitKey(500)
#         # cv2.destroyAllWindows()
#     return images_augmentation
#
#
# # path = 'tryyyyyyyyyyyy.jpg'
# # path2 = 'image_augmantion'
# #
# #
# # image_augmentation(path, 1, path2)
#
# # function that return image whith white bakeground for size 224/224 to sent the model
# def white(img):
#     # read image
#     ht, wd, cc = img.shape
#     # create new image of desired size and color (white) for padding
#     ww = 224
#     hh = 224
#     color = (255, 255, 255)
#     result = numpy.full((hh, ww, cc), color, dtype=numpy.uint8)
#     # set offsets for top left corner
#     xx = 0
#     yy = 0
#     # copy img image into center of result image
#     result[yy:yy + ht, xx:xx + wd] = img
#     return result
#
# # function that get image and place and return fix image for the model
# # the place mast be: [x,y,w,h]
def cut_place(image,place):

    placeInt=place[1:-2]
    placeInt=placeInt.split(',')
    x=round(float(placeInt[0]))
    y=round(float(placeInt[1]))
    w = round(float(placeInt[2]))
    h = round(float(placeInt[3]))
    count=h*w
    print(count, "pixels")
    if(count>50,176):
        crop_img = image[y:y + h, x:x + w]
        crop_img =changeResolution2(crop_img)
    if(count<50,176):
        if(h<224 and w<224):
            crop_img = image[y:y + h, x:x + w]
            crop_img=white(crop_img)
        else:
            if(h<224):
                h=224
            if(w<224):
                w=224
            crop_img = image[y:y + h, x:x + w]
            crop_img = changeResolution2(crop_img)
    else:
        crop_img = image[y:y + h, x:x + w]

    height,width,c=crop_img.shape
    print(height*width, "pixels")
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(500)
    # cv2.destroyAllWindows()
    return crop_img
#
# # פונקציה שמקבלת מערך של תמונות עם מיקום של אוביקטים ומחזירה מערך עם האוביקטים לשליחה למודל
# def buf_cut_places(buf):
#     for image in buf:
#         for object in image._objects:
#             object._objectC=cut_place(image,object._placeC)
#     return buf
#
# # function that get csv whith path, and place for every image, and directory for the fix images
# def create_dataset_from_csv(csv, frame_dir: str,name="image", ext="jpg"):
#     #create dir for the crope images in this csv
#     v_name = splitext(basename(csv))[0]
#     video_path_arr = csv.split('/')
#     print(video_path_arr)
#     if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
#         frame_dir = dirname(frame_dir)
#     frame_dir_ = join(frame_dir, v_name)
#
#     makedirs(frame_dir_, exist_ok=True)
#     base_path = join(frame_dir_, name)
#     print(base_path)
#
#     df = pd.read_csv(csv)
#     numOfImages = 1
#     for i in range(0,df.shape[0],17):
#         filePath = df['imageFilename'][i]
#         filePath = filePath.replace('\\', '/')
#         filePath = filePath.replace('\'', '')
#         print(filePath)
#         place=''
#         if df['BIRD'][i]!='[]':
#             place = df['BIRD'][i]
#         if df['AIRPLANE'][i] != '[]':
#             place = df['AIRPLANE'][i]
#         if df['DRONE'][i]!='[]':
#             place = df['DRONE'][i]
#         if df['HELICOPTER'][i]!='[]':
#             place = df['HELICOPTER'][i]
#         if place[0]=='[':
#             print(place)
#             image = cv2.imread(filePath, 1)
#             if place.find(';')!=-1:
#                 newPlace=place.split(';')
#                 a=newPlace[0]+']'
#                 b='['+newPlace[1]
#                 img1 = cut_place(image, a)
#                 img=[]
#                 if(img1!=[]):
#                     img.append(img1)
#                 img2 = cut_place(image, b)
#                 if(img2!=[]):
#                     img.append(img2)
#             else:
#                 img=[]
#                 img1=cut_place(image, place)
#                 if(img1!=[]):
#                     img.append(img1)
#             #copy the crope image to the directory
#             if numOfImages == 1:  #Save 0 second frame
#                 for i in img:
#                     cv2.imwrite("{}_{}.{}".format(base_path, "0000", ext), i)
#                     numOfImages += 1
#             else:
#                 for i in img:
#                     filled_numOfImages = str(numOfImages).zfill(4)
#                     cv2.imwrite("{}_{}.{}".format(base_path, filled_numOfImages, ext),i)
#                     print("{}_{}.{}".format(base_path, filled_numOfImages, ext))
#                     numOfImages += 1
#     print(numOfImages)
#
#
#
# # create the dataset
# path=['airplain_close_csv','airplain_medium_csv','bird_close_csv','bird_medium_csv','drone_close_csv','drone_distance_csv','drone_medium1_csv','drone_medium2_csv','helicopter_close_csv','helicopter_medium_csv']
# for i in path:
#     csvPath=i+'.csv'
#     name_dir='./create_dataset2'
#     create_dataset_from_csv(csvPath,name_dir)
#
#
# # filePath='BIRD_MEDIUM\V_BIRD_04621_311.png'
# # image = cv2.imread(filePath, 1)
# # place="[125,181,22,29]"
# # detect_face(image,place)
# # filePath='resized_image1.jpg'
# # image = cv2.imread(filePath, 1)
# # height, width, c = image.shape
# # print(height * width, "pixels")







