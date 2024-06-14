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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ARCHIVE_FOLDER'] = 'archives'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ARCHIVE_FOLDER'], exist_ok=True)

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x):
    local_yolo = YoloModel()
    yolo_output = local_yolo.video_detection(path_x)  # generator

    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return 'last_frame'

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
        file.save(save_path)

        result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], dir_name, 'results', filename)

        session['video_path'] = save_path
        session['result_file_path'] = result_file_path
    return render_template('videoprojectnew.html', form=form)

@app.route('/video')
def video():
    video_path = session.get('video_path', None)
    result_file_path = session.get('result_file_path', None)
    if video_path:
        frame = generate_frames(video_path)
        if frame != 'last_frame':
            return Response(frame, mimetype='multipart/x-mixed-replace; boundary=frame')

        else:
            print('DEBUG last_frame')
            is_generated = True
            return render_template('videoprojectnew.html', is_generated=is_generated, filename=result_file_path)

    return jsonify({"error": "No video path in session"}), 400

@app.route('/save_results', methods=['POST'])
def save_results():
    video_path = session.get('video_path', None)
    if video_path:
        filename = os.path.basename(video_path)
        dir_name, _ = os.path.splitext(filename)
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], dir_name, 'results')
        zip_file_path = os.path.join(app.config['ARCHIVE_FOLDER'], dir_name)
        shutil.make_archive(base_name=str(zip_file_path), format='zip', root_dir=str(folder_path))
        return send_file(path_or_file=f'{zip_file_path}.zip', as_attachment=True)

    return jsonify({"error": "No video path in session"}), 400

@app.route('/webapp')
def webapp():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
