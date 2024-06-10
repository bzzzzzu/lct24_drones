import os
import numpy as np
import cv2
from flask import Flask, render_template, request, Response, session, redirect, url_for
from flask_socketio import SocketIO
import yt_dlp as youtube_dl
from yolo_inference import YoloContainerInference
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'detects'
os.makedirs(app.config['UPLOAD_FOLDER']) if not os.path.exists(app.config['UPLOAD_FOLDER']) else None
socketio = SocketIO(app, async_mode='threading')
stop_flag = False

yolo_inference = YoloContainerInference()


class YoloWebService(object):
    def __init__(self):
        super(YoloWebService, self).__init__()
        print("*********************************WEB SERVICE START******************************")
        self._preview = False
        self._flipH = False
        self._detect = False
        self._confidence = 75.0

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        self._confidence = int(value)

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)

    def start_stream(self, url):
        self._preview = False
        self._flipH = False
        self._detect = False
        self._confidence = 75.0

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best",
            "forceurl": True,
        }
        ydl = youtube_dl.YoutubeDL(ydl_opts)
        info = ydl.extract_info(url, download=False)
        url = info["url"]
        cap = cv2.VideoCapture(url)

        while True:
            if self._preview:
                if stop_flag:
                    print("Process Stopped")
                    return

                grabbed, frame = cap.read()
                if not grabbed:
                    break
                if self.flipH:
                    frame = cv2.flip(frame, 1)
                if self.detect:
                    print(self._confidence)
                    results = yolo_inference.detect(frame, stream=True, name='stream', conf=self._confidence/100)

                    for result in results:
                        frame = result.orig_img
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)

                        for box, score, class_id in zip(boxes, confidences, class_ids):
                            x1, y1, x2, y2 = map(int, box)
                            label = f"{yolo_inference.class_names[class_id]}: {score:.2f}"
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                frame = cv2.imencode(".jpg", frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                snap = np.zeros((1000, 1000), np.uint8)
                label = "Streaming Off"
                H, W = snap.shape
                font = cv2.FONT_HERSHEY_PLAIN
                color = (255, 255, 255)
                cv2.putText(snap, label, (W // 2 - 100, H // 2), font, 2, color, 2)
                frame = cv2.imencode(".jpg", snap)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def start_images(self, file_path):
        image = cv2.imread(file_path)
        results = yolo_inference.detect(image, stream=False, name='images', conf=self._confidence/100)
        for result in results:
            _, buffer = cv2.imencode('.jpg', result.orig_img)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
        return encoded_image

    def start_videos(self, file_path):
        cap = cv2.VideoCapture(file_path)
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            results = yolo_inference.detect(frame, stream=False, name='video', conf=self._confidence/100)
            for result in results:
                _, buffer = cv2.imencode('.jpg', result.orig_img)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                frames.append(encoded_image)

        cap.release()
        return frames


WEBS = YoloWebService()


@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/streaming', methods=['GET', 'POST'])
def streaming():
    global stop_flag
    stop_flag = False
    if request.method == 'POST':
        url = request.form['url']
        session['url'] = url
        return redirect(url_for('video_feed'))
    return render_template('streaming.html')


@app.route('/video_feed')
def video_feed():
    url = session.get('url', None)
    if url is None:
        print('url is None')
        return redirect(url_for('homepage'))
    return Response(WEBS.start_stream(url), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/request_preview_switch")
def request_preview_switch():
    WEBS.preview = not WEBS.preview
    return "nothing"


@app.route("/request_flipH_switch")
def request_flipH_switch():
    WEBS.flipH = not WEBS.flipH
    return "nothing"


@app.route("/request_run_model_switch")
def request_run_model_switch():
    WEBS.detect = not WEBS.detect
    return "nothing"


@app.route('/update_slider_value', methods=['POST'])
def update_slider_value():
    slider_value = request.form['sliderValue']
    WEBS.confidence = slider_value
    return 'OK'


@app.route('/stop_process')
def stop_process():
    global stop_flag
    stop_flag = True
    return 'Process Stop Request'


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if not file or file.filename == '':
            return redirect(request.url)

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            processed_image = WEBS.start_images(file_path)
            return render_template('upload.html', image=processed_image)
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            processed_frames = WEBS.start_videos(file_path)
            return render_template('upload.html', frames=processed_frames)

    return render_template('upload.html')


if __name__ == "__main__":
    # socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

