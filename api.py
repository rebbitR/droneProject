
from flask import Flask, request
from flask_cors import CORS
from main import main
from video import frames_to_video
from log import create_log

app = Flask(__name__)
CORS(app)

@app.route('/post', methods=['POST'])
def program():
    create_log()
    video_path="video_from_client/myvideo_1.mp4"
    res_video_path='videoNew_1.mp4'

    file = request.files['myFile']
    file.save(video_path)

    buf=main(video_path)
    frames_to_video(res_video_path,buf)

    return 'http://127.0.0.1:8887/'+'videoNew_1.mp4'


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)










