import cv2
import random
import numpy as np
import os
from flask import Flask, request, jsonify, json
from flask_cors import CORS

import main



from os.path import isfile, join
#להמיר ודיאו למסגרות-תמונות
#Convert video to frames
# def videotoframes(video):
#  vidcap = cv2.VideoCapture(video)
#
#  list=[]
#  def getFrame(sec):
#    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#    hasFrames,image = vidcap.read()
#
#    if hasFrames:
#     list.append(image)
#
#     #cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
#     return hasFrames
#  sec = 0
#  frameRate = 0.08 #//it will capture image in each 0.5 second
#  count=1
#  success = getFrame(sec)
#  while success:
#    count = count + 1
#    sec = sec + frameRate
#    sec = round(sec, 2)
#    success = getFrame(sec)
#  return list



#ליצור ודיאו מתמונות
#Create video from python images
def framestovideo(list=[]):
  image_folder = 'imeges'
  video_name = 'M:/myproject/react/lesson13-upload/src/video/videoneww.mp4'

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



app = Flask(__name__)

CORS(app)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.route('/post', methods=['POST'])
def json_example():





    file = request.files['myFile']
    file.save("M:/myproject/react/lesson13-upload/src/video/myvideo.mp4")
   # file = request_data['file']
   #
    arr = main.program("M:/myproject/react/lesson13-upload/src/video/myvideo.mp4")
    # for image in arr:
    #     x, y, w, h = model(image)
    #     my(image, x, y, w, h)

    framestovideo(arr)
    filename1="videoneww.mp4"
    url1="M:/myproject/react/lesson13-upload/src/video/videoneww.mp4"


    return jsonify({'filename': filename1,'url':url1})


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)









