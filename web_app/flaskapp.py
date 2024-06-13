from flask import Flask, render_template, Response, jsonify, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from yolo_detect import YoloModel
from threading import Lock

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

executor = ThreadPoolExecutor(2)
lock = Lock()

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x=''):
    with lock:
        local_yolo = YoloModel()
        yolo_output = local_yolo.video_detection(path_x)
        for detection_ in yolo_output:
            ref, buffer = cv2.imencode('.jpg', detection_)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_web(path_x):
    with lock:
        local_yolo = YoloModel()
        yolo_output = local_yolo.video_detection(path_x)
        for detection_ in yolo_output:
            ref, buffer = cv2.imencode('.jpg', detection_)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')

@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                 secure_filename(file.filename))
        file.save(save_path)
        session['video_path'] = save_path
    return render_template('videoprojectnew.html', form=form)

@app.route('/video')
def video():
    video_path = session.get('video_path', None)
    if video_path:
        executor.submit(generate_frames, video_path)
        return Response(generate_frames(video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return jsonify({"error": "No video path in session"}), 400

@app.route('/webapp')
def webapp():
    executor.submit(generate_frames_web, 0)
    return Response(generate_frames_web(0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
