
from flask import Flask, request, jsonify
from flask_cors import CORS
from main import main
from video import frames_to_video
from log import CreateLog

app = Flask(__name__)
CORS(app)

@app.route('/post', methods=['POST'])
def program():
    print('i am in program')
    CreateLog()
    video_path="D:/react/src/video/myvideo_1.mp4"
    res_video_path='videoNew_1.mp4'
    # video_path=r'C:\Users\רננה קייקוב\Desktop\Project\droneProject-client\react\src\video\myvideo_1.mp4'
    # res_video_path=r'C:\Users\רננה קייקוב\Desktop\Project\droneProject-client\react\src\video\videoneww_1.mp4'

    file = request.files['myFile']
    file.save(video_path)

    buf=main(video_path)
    frames_to_video(res_video_path,buf)

    filename1="videoneww_1.mp4"
    return 'http://127.0.0.1:8887/'+'videoNew_1.mp4'
    # url1=res_video_path
    # describe=['detection drone']
    #
    # return jsonify({'filename': filename1,'url':url1,'dis':describe})

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)










