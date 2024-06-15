import json
import time
from typing import List

from flask import Flask, render_template, Response, jsonify, request, session, send_file
from flask_wtf import FlaskForm
from wtforms import MultipleFileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.fields.simple import StringField, FileField
from wtforms.validators import InputRequired, ValidationError, DataRequired
import os
import cv2
from threading import Thread

from values import VIDEO_EXTENSIONS, IMG_EXTENSIONS
from yolo_detect import YoloModel
import shutil
from datetime import datetime
import tempfile

app = Flask(__name__)
local_yolo = YoloModel()

app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class FileFolderForm(FlaskForm):
    file_folder = FileField('Select Files or Folder', validators=[DataRequired()])
    submit = SubmitField('Обработать')


def generate_frames(path_x: List[str]):
    print(f'DEBUG: {path_x=}')
    current_datetime = str(datetime.now())
    for path_ in path_x:
        yolo_output = local_yolo.video_detection(path_, current_datetime)  # return generator

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
    form = FileFolderForm()

    def process_file(file, folder=None):
        filename = secure_filename(file.filename)
        if folder:
            filename = filename.replace(folder + "_", "")
        full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dir_name = os.path.basename(full_filepath)
        dir_name, _ = os.path.splitext(dir_name)
        full_dirpath = os.path.join(app.config['UPLOAD_FOLDER'], dir_name)
        os.makedirs(full_dirpath, exist_ok=True)
        os.makedirs(os.path.join(full_dirpath, 'original'), exist_ok=True)
        save_path = os.path.join(full_dirpath, 'original', filename)
        print(save_path, not os.path.exists(save_path))
        if not os.path.exists(save_path):
            file.save(save_path)
            if not session.get('video_paths', None):
                session['video_paths'] = []
            session['video_paths'].append(save_path)

        else:
            if not session.get('video_paths', None):
                session['video_paths'] = []

        result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], dir_name, 'results', filename)

        if session.get('results_file_names', None) is None:
            session['results_file_names'] = []
        session['results_file_names'].append(filename)

        session['result_file_path'] = result_file_path

    if form.validate_on_submit():
        session['results_file_names'] = None
        session['video_paths'] = None
        session['result_file_path'] = None
        file_folder_data = request.files.getlist('file_folder')
        if file_folder_data and len(file_folder_data) > 1:
            for file in file_folder_data:
                folder = "_".join(file.filename.split('/')[:-1])
                process_file(file, folder)
        else:
            process_file(file_folder_data[0])

    return render_template('videoprojectnew.html', form=form)


@app.route('/frames')
def frames():
    video_paths = session.get('video_paths', None)
    print(f"{video_paths=}")

    if video_paths:
        frame = generate_frames(video_paths)
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


@app.route('/check_status')
def check_status():
    result_file_path = session.get('result_file_path', None)
    print(f"{result_file_path=}")

    if result_file_path and result_file_path.endswith(VIDEO_EXTENSIONS):
        timecode_file_path = result_file_path.replace('/results/', '/timecodes/')
        timecode_file_path = timecode_file_path.replace('.mp4', '.json')

        if os.path.exists(timecode_file_path) and os.path.exists(result_file_path):
            return jsonify({'status': 'ready', 'result_file_path': result_file_path, "type": "video"})
    elif result_file_path and result_file_path.endswith(IMG_EXTENSIONS) and os.path.exists(result_file_path):
        return jsonify({'status': 'ready', 'result_file_path': result_file_path, "type": "image"})

    return jsonify({'status': 'processing', "type": None})


@app.route('/video/<path:result_file_path>')
def video(result_file_path):
    if result_file_path and os.path.exists(result_file_path):
        print("result_file_path", result_file_path)
        return send_file(result_file_path, mimetype='video/mp4')
    return jsonify({"error": "No video found"}), 404


@app.route('/image/<path:result_file_path>')
def image(result_file_path):
    if result_file_path and os.path.exists(result_file_path):
        return send_file(result_file_path)
    return jsonify({"error": "No image found"}), 404


@app.route('/save_results', methods=['POST'])
def save_results():
    results_video_names = session['results_file_names']
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_common_folder_path = os.path.join(temp_dir, "common")
        for filename in results_video_names:
            dir_name, _ = os.path.splitext(filename)
            folder_path = os.path.join(app.config['UPLOAD_FOLDER'], dir_name, 'results')

            temp_folder_path = os.path.join(temp_common_folder_path, dir_name)
            shutil.copytree(folder_path, temp_folder_path)
        shutil.make_archive(base_name=temp_common_folder_path, format='zip', root_dir=temp_common_folder_path)
        return send_file(path_or_file=temp_common_folder_path + ".zip", as_attachment=True)

    return jsonify({"error": "No video path in session"}), 400


@app.route('/webapp')
def webapp():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
