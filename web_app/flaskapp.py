import json

from flask import Flask, render_template, Response, jsonify, request, session, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
from threading import Thread
from yolo_detect import YoloModel
import shutil
from datetime import datetime
import tempfile

app = Flask(__name__)
local_yolo = YoloModel()

app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x):
    print(f'DEBUG: {path_x=}')
    current_datetime = str(datetime.now())
    yolo_output = local_yolo.video_detection(path_x, current_datetime)  # return generator

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
        filename = secure_filename(file.filename)

        full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dir_name = os.path.basename(full_filepath)
        dir_name, _ = os.path.splitext(dir_name)
        full_dirpath = os.path.join(app.config['UPLOAD_FOLDER'], dir_name)
        os.makedirs(full_dirpath, exist_ok=True)
        os.makedirs(os.path.join(full_dirpath, 'original'), exist_ok=True)
        save_path = os.path.join(full_dirpath, 'original', filename)
        if not os.path.exists(save_path):
            file.save(save_path)
            session['video_path'] = save_path
        else:
            session['video_path'] = None
        result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], dir_name, 'results',
                                        filename)
        session['result_file_path'] = result_file_path
    return render_template('videoprojectnew.html', form=form)


@app.route('/frames')
def frames():
    video_path = session.get('video_path', None)
    if video_path:
        frame = generate_frames(video_path)
        return Response(frame, mimetype='multipart/x-mixed-replace; boundary=frame')
    return jsonify({"error": "No video path in session"}), 400


@app.route('/timecodes/<path:result_file_path>')
def timecodes(result_file_path):
    timecodes_path = result_file_path.replace("results", "timecodes")
    timecodes_path = ".".join(timecodes_path.split(".")[:-1]) + ".json"
    if timecodes_path and os.path.exists(timecodes_path):
        with open(timecodes_path, "r") as f:
            timecodes = json.load(f)
        return jsonify(timecodes), 200
    return jsonify({"error": "No timecodes path in session"}), 400


@app.route('/check_video_status')
def check_video_status():
    # FIXME: Файл создается и сохраняется во время работы модели. Т.е. файл уже существует
    # FIXME: Тут надо проверять на завершение работы модели/ наличие файла timecodes

    result_file_path = session.get('result_file_path')
    timecode_file_path = result_file_path.replace('/results/', '/timecodes/')
    timecode_file_path = timecode_file_path.replace('.mp4', '.json')

    if os.path.exists(timecode_file_path) and os.path.exists(result_file_path):
        return jsonify({'status': 'ready', 'result_file_path': result_file_path})
    else:
        return jsonify({'status': 'processing'})


@app.route('/video/<path:result_file_path>')
def video(result_file_path):
    if result_file_path and os.path.exists(result_file_path):
        print("result_file_path", result_file_path)
        return send_file(result_file_path, mimetype='video/mp4')
    return jsonify({"error": "No video found"}), 404


@app.route('/save_results', methods=['POST'])
def save_results():
    video_path = session.get('result_file_path', None)
    if video_path:
        filename = os.path.basename(video_path)
        dir_name, _ = os.path.splitext(filename)
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], dir_name, 'results')

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_folder_path = os.path.join(temp_dir, dir_name)
            shutil.copytree(folder_path, temp_folder_path)
            zip_file_path = os.path.join(temp_dir, f'{dir_name}.zip')
            shutil.make_archive(base_name=temp_folder_path, format='zip', root_dir=temp_folder_path)
            return send_file(path_or_file=zip_file_path, as_attachment=True)

    return jsonify({"error": "No video path in session"}), 400


@app.route('/webapp')
def webapp():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
