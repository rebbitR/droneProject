import cv2
import random
import numpy as np
import os
from flask import Flask, request, jsonify, json
from flask_cors import CORS
# import face_recognition as fr
from classes import frame
# import moviepy.editor as mp

from os.path import isfile, join
#להמיר ודיאו למסגרות-תמונות
#Convert video to frames
def videotoframes(video):
 print('I am in videotoframes')
 vidcap = cv2.VideoCapture(video)
#שמירת השמע
 #clip = mp.VideoFileClip(video).subclip(0, 20)
 #clip.audio.write_audiofile(r"my_result.mp3")
 list=[]
 def getFrame(sec):
   vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
   hasFrames,image = vidcap.read()

   if hasFrames:
    list.append(image)

    #cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
 sec = 0
 frameRate = 0.08 #//it will capture image in each 0.5 second
 count=1
 success = getFrame(sec)
 while success:
   count = count + 1
   sec = sec + frameRate
   sec = round(sec, 2)
   success = getFrame(sec)
 return list

#טישטוש חלקים מתמונה
# Blurring parts of a picture in a python
#def my(image,a):
def my(image,a):
    for i in a:
        x=i[0]
        y=i[1]
        w=i[2]
        h=i[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y + h, x:x + w]
        # applying a gaussian blur over this new rectangle area
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        # impose this blurred image on original image to get final image
        image[y:y + roi.shape[0], x:x + roi.shape[1]] = roi
    return image
      #for i in a:
        # cv2.rectangle(image, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0, 255, 0), 2)
        # roi = image[i[1]:i[1] + i[3], i[0]:i[0] + i[2]]
        # # applying a gaussian blur over this new rectangle area
        # roi = cv2.GaussianBlur(roi, (23, 23), 30)
        # # impose this blurred image on original image to get final image
        # image[i[1]:i[1] + roi.shape[0], i[0]:i[0] + roi.shape[1]] = roi


#ליצור ודיאו מתמונות
#Create video from python images
def framestovideo(list=[]):
  image_folder = 'imeges'
  video_name = 'C:/Users/רננה קייקוב/Desktop/Project/פרויקט זיהוי רחפן/react/src/video/videoneww_1.mp4'

  #images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
  #frame = cv2.imread(os.path.join(image_folder, list[0]))
  #frame = cv2.imread( list[0])

  #height, width, layers = frame.shape
  height, width, layers = list[0].shape
  video = cv2.VideoWriter(video_name, 0, 12, (width,height))

  for image in list:
    #video.write(cv2.imread(os.path.join(image_folder, image)))
    video.write(image)

  cv2.destroyAllWindows()

  video.release()
  #הוספת שמע לסרטון
  #adiuo=cv2.
  #ffmpeg.output(audio_stream, video, 'out.mp4').run()



def model(image):
    # encoding = fr.face_encodings(image)[0]
    # print(encoding)
    #return(random.randint(0, 300),random.randint(0,300),random.randint(10, 200),random.randint(10, 200))
    a=[]
    b=[]
    x = random.randint(150, 200)
    x1 = random.randint(150, 200)
    x2 = random.randint(150, 200)
    x3 = random.randint(150, 200)
    b.append(x)
    b.append(x1)
    b.append(x2)
    b.append(x3)
    a.append(b)
    #return (150,200,150,200)
    return (a)
    #return (x,x1,x2,x3)


# arr=videotoframes('sirton.mp4')
# for image in arr:
#     x,y,w,h=model(image)
#     my(image,x,y,w,h)
#
# framestovideo(arr)


#רשימה לטקסט
def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 =str1+ele

        # return string
    return str1







app = Flask(__name__)

CORS(app)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.route('/post', methods=['POST'])
def json_example():
    # if 'myFile' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)
    # file = request.files['myFile']
    # # if user does not select file, browser also
    # # submit a empty part without filename
    # if file.filename == '':
    #     flash('No selected file')
    #     return redirect(request.url)
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #     vidoe = upload(open(filename, 'rb'))
    #     #user_id = user.insert({"file": image})
    #     return jsonify({'result': "E:\כאן שומרים\מיכל\ppppp\PROJECT\videoneww.mp4"})
    #
    #





    file = request.files['myFile']
    file.save("C:/Users/רננה קייקוב/Desktop/Project/פרויקט זיהוי רחפן/react/src/video/myvideo_1.mp4")
   # file = request_data['file']

    arr = videotoframes("C:/Users/רננה קייקוב/Desktop/Project/פרויקט זיהוי רחפן/react/src/video/myvideo_1.mp4")
    # #תיאור בכל פרם כמה פרצופים נמצאו
    # discrib=[]
    # for image in arr:
    #
    #     a = model(image)
    #     b="sumfaice:"+str(len(a))
    #     discrib.append(b)
    #     my(image, a)
    #     #,x1,x2,x3 = model(image)
    #     #my(image,x,x1,x2,x3)
    print('I am in json_example')
    buf=[]
    second=0
    for image in arr:
        myf1 = frame(image, second)
        buf.append(myf1)
        second=second+1
    for frame1 in buf:
        # find_objects at frame _with_yolo:
        frame1.yolo_detect()
        # cut_objects_from_frame():
        frame1.cut_objects()
        # find_kinds_with_model:
        frame1.model('category')
        frame1.result()
    arr_result=[]
    for frame2 in buf:
        arr_result.append(frame2.frameC)


    framestovideo(arr_result)
    filename1="videoneww_1.mp4"
    url1="C:/Users/רננה קייקוב/Desktop/Project/פרויקט זיהוי רחפן/react/src/video/videoneww_1.mp4"
    describe=['ren']

    return jsonify({'filename': filename1,'url':url1,'dis':describe})


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)









