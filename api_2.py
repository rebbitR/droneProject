import cv2
import random
import numpy as np
import os
from flask import Flask, request, jsonify, json
from flask_cors import CORS
# import face_recognition as fr
from classes import frame
# import moviepy.editor as mp
from main import main
from os.path import isfile, join
from cut_video import framestovideo




app = Flask(__name__)

CORS(app)




@app.route('/post', methods=['POST'])
def program():

    video_path="D:/react/src/video/myvideo_1.mp4"
    res_video_path='D:/react/src/video/videoneww_1.mp4'

    file = request.files['myFile']
    file.save(video_path)

    buf=main(video_path)

    buf1=[]
    for i in buf:
        buf1.append(i.frameC)
    framestovideo(res_video_path,buf1)

    filename1="videoneww_1.mp4"
    url1=res_video_path
    describe=['detection drone']

    return jsonify({'filename': filename1,'url':url1,'dis':describe})


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)









