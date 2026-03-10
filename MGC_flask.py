from flask import Flask, render_template, request
import smtplib
from flask_mysqldb import MySQL
from datetime import datetime
import tensorflow as tf
import keras
import librosa
import numpy as np
import math
import os
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from pydub.utils import which

# Set ffmpeg path for pydub
AudioSegment.converter = which(
    r"C:/Users/vishalvk/OneDrive/Desktop/Music-Genre-Classification-main/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"
)

app = Flask(__name__)

# Configure MySQL (Update these values for your database)
app.config['MYSQL_HOST'] = "YOUR_HOST"
app.config['MYSQL_USER'] = "YOUR_USERNAME"
app.config['MYSQL_PASSWORD'] = "YOUR_PASSWORD"
app.config['MYSQL_DB'] = "YOUR_DB_NAME"
app.config['MYSQL_CURSORCLASS'] = "DictCursor"

mysql = MySQL(app)

# Load the trained CNN model
model = keras.models.load_model("MusicGenre_CNN_79.73.h5")

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def homepage():
    return render_template('homepage.html', title="MGC")

@app.route("/prediction", methods=["POST"])
def prediction():
    title = "MGC | Prediction"

    if 'myfile' not in request.files:
        return "No file part", 400

    file = request.files['myfile']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Convert MP3 to WAV if needed and trim 60s to 90s
    if filename.lower().endswith(".mp3"):
        audio_segment = AudioSegment.from_mp3(file_path)
        trimmed_audio = audio_segment[60 * 1000:90 * 1000]
        wav_filename = filename.rsplit('.', 1)[0] + ".wav"
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
        trimmed_audio.export(wav_path, format="wav")
    else:
        wav_path = file_path  # Assume it's WAV already

    # Audio preprocessing
    def process_input(audio_file, track_duration=30):
        SAMPLE_RATE = 22050
        NUM_MFCC = 13
        N_FTT = 2048
        HOP_LENGTH = 512
        SAMPLES_PER_TRACK = SAMPLE_RATE * track_duration
        NUM_SEGMENTS = 10

        samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

        signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
        mfccs = []

        for d in range(NUM_SEGMENTS):
            start = samples_per_segment * d
            finish = start + samples_per_segment
            mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate,
                                        n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
            mfcc = mfcc.T
            if len(mfcc) == num_mfcc_vectors_per_segment:
                mfccs.append(mfcc)

        if not mfccs:
            return None
        return np.array(mfccs[0])  # use the first valid segment

    mfcc_features = process_input(wav_path)

    if mfcc_features is None:
        return "Could not extract audio features", 500

    X_to_predict = mfcc_features[np.newaxis, ..., np.newaxis]
    predictions = model.predict(X_to_predict)
    pred_index = np.argmax(predictions)
    probabilities = predictions[0]

    genre_dict = {
        0: "disco", 1: "pop", 2: "classical", 3: "metal", 4: "rock",
        5: "blues", 6: "hiphop", 7: "reggae", 8: "country", 9: "jazz"
    }

    # Top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]

    return render_template('prediction.html',
                           title=title,
                           prediction=genre_dict[pred_index],
                           probability="{:.2f}".format(probabilities[pred_index] * 100),
                           second_prediction=genre_dict[top_indices[1]],
                           second_probability="{:.2f}".format(probabilities[top_indices[1]] * 100),
                           third_prediction=genre_dict[top_indices[2]],
                           third_probability="{:.2f}".format(probabilities[top_indices[2]] * 100))

@app.route("/about")
def about():
    return render_template('about.html', title="MGC | About")

@app.route("/project")
def project():
    return render_template('project.html', title="MGC | Project")

@app.route("/contact")
def contact():
    return render_template('contact.html', title="MGC | Contact")

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'GET':
        return "Login via the login form"

    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        phone_number = request.form['phone_number']
        url = request.form['url']
        message = request.form['message']
        time = datetime.now()

        # SMTP Email Sending (Gmail)
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login("your_email_address", "your_password")  # Use env vars in production
        server.sendmail("your_email_address", email, message)
        server.quit()

        # Save to MySQL
        cursor = mysql.connection.cursor()
        cursor.execute('''INSERT INTO Contacts(full_name, email, phone_number, url, message, time)
                          VALUES(%s, %s, %s, %s, %s, %s)''',
                       (full_name, email, phone_number, url, message, time))
        mysql.connection.commit()
        cursor.close()

        return render_template('contact.html', title="MGC | Contact")

# ✅ Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
