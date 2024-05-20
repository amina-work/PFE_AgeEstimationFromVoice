from flask import Flask, render_template_string, request, jsonify
import librosa
import numpy as np
from keras.models import load_model
import os
import random
from pydub import AudioSegment
AudioSegment.ffmpeg = r"C:\ffmpeg\ffmpeg-2024-05-20-git-127ded5078-full_build\bin\ffmpeg.exe"

app = Flask(__name__)


sentences = [
    "most of them were staring quietly at the big table",
    "you point to a trail of ants leading into the house",
    "the time must have been somewhere around six o'clock",
    "as i watched the planet seemed to grow larger and smaller",
    "i have the diet of a kid who found twenty dollars"
]

age_classes = ["> 19", "20 - 29", "30-39", "40-49", "50-59", "60-69", "70-79", "80 <"]
model = load_model('my_model.h5')

@app.route('/')
def index():
    sentence = random.choice(sentences)

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Voice-Based Age Estimation</title>
        <link rel="stylesheet" href="static/app.css">
    </head>
    <body>
        <div class="container" id="questions">
            <h1>Voice-Based Age Estimation</h1>
            <form id="ageEstimationForm" action="/predict" method="POST" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="gender">Gender:</label>
                    <select id="gender" name="gender">
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                        <option value="other">Other</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="accent">Accent:</label>
                    <select id="accent" name="accent">
                        <option value="us">US</option>
                        <option value="england">England</option>
                        <option value="indian">Indian</option>
                        <option value="australia">Australia</option>
                        <option value="canada">Canada</option>
                        <option value="scotland">Scotland</option>
                        <option value="african">African</option>
                        <option value="newzealand">New Zealand</option>
                        <option value="ireland">Ireland</option>
                        <option value="philippines">Philippines</option>
                        <option value="wales">Wales</option>
                        <option value="bermuda">Bermuda</option>
                        <option value="malaysia">Malaysia</option>
                        <option value="singapore">Singapore</option>
                        <option value="hongkong">Hong Kong</option>
                        <option value="southatlandtic">South Atlantic</option>
                    </select>
                </div>
                <div class="sentence">
                    <h4>Read out this sentence please:</h4>
                    <p id="sentence">{{ sentence }}</p>
                </div>
                <div class="form-group">
                    <label for="audioFile">Upload an audio recording of your voice:</label>
                    <input type="file" id="audioFile" name="audioFile" accept="audio/mp3">
                </div>
                <div class="form-group">
                    <label for="recordAudio">Or record an audio recording of your voice:</label>
                    <div class="recording-btns">
                        <audio id="audioPlayback" controls></audio>
                        <button type="button" id="recordButton">Record</button>
                        <button type="button" id="stopButton" disabled>Stop</button>
                    </div>
                    <input type="hidden" id="recordedAudioPath" name="recordedAudioPath">
                </div>
                <button type="submit">Submit</button>
            </form>
        </div>
        <div class="container" id="results">
            <h1>The results</h1>
            <div id="ageResult"></div>
            <div id="ageResultProbability"></div>
            <button id="tryAgainButton">Try Again?</button>
        </div>
        <script>
            document.getElementById('ageEstimationForm').addEventListener('submit', function(event) {
                event.preventDefault();
                var formData = new FormData(this);
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('questions').style.display = 'none';
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('ageResult').textContent = 'Age Category: ' + data.top_predicted_age_category.age_category;
                    document.getElementById('ageResultProbability').textContent = 'Probability: ' + (data.top_predicted_age_category.probability * 100).toFixed(2) + '%';
                })
                .catch(error => console.error('Error:', error));
            });
            document.getElementById('tryAgainButton').addEventListener('click', function() {
                document.getElementById('results').style.display = 'none';
                document.getElementById('questions').style.display = 'block';
            });

            let mediaRecorder;
            let recordedChunks = [];

            document.getElementById('recordButton').addEventListener('click', function() {
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(stream => {
                        mediaRecorder = new MediaRecorder(stream);
                        mediaRecorder.start();
                        mediaRecorder.ondataavailable = function(event) {
                            if (event.data.size > 0) {
                                recordedChunks.push(event.data);
                            }
                        };
                        mediaRecorder.onstop = function() {
                            const blob = new Blob(recordedChunks, { type: "audio/mp3" });
                            recordedChunks = [];
                            const audioURL = URL.createObjectURL(blob);
                            document.getElementById('audioPlayback').src = audioURL;
                            document.getElementById('audioPlayback').play();
                        };
                        document.getElementById('recordButton').disabled = true;
                        document.getElementById('stopButton').disabled = false;
                    })
                    .catch(error => console.error('Error accessing microphone:', error));
            });

            document.getElementById('stopButton').addEventListener('click', function() {
                mediaRecorder.stop();
                document.getElementById('recordButton').disabled = false;
                document.getElementById('stopButton').disabled = true;
                document.getElementById('recordedAudioPath').value = audioURL;
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content, sentence=sentence)

@app.route('/predict', methods=['POST'])
def predict_age():
    if 'audioFile' in request.files:
        file = request.files['audioFile']
        audio_path = os.path.join('newdata', 'temp_audio.mp3')
        file.save(audio_path)
    elif 'recordedAudioPath' in request.form:
        audio_path = request.form['recordedAudioPath']
    else:
        return jsonify({'error': 'No audio file uploaded or recorded'})

    gender = request.form.get('gender')
    accent = request.form.get('accent')

    # Convert gender and accent to one-hot encoded arrays
    gender_encoding = {'female': [0, 1, 0], 'male': [1, 0, 0], 'other': [0, 0, 1]}
    accent_encoding = {
        'us': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'england': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'indian': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'australia': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'canada': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'scotland': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'african': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'newzealand': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'ireland': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'philippines': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'wales': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'bermuda': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'malaysia': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'singapore': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'hongkong': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'southatlandtic': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }

    gender = gender_encoding.get(gender, [0, 0, 0])
    accent = accent_encoding.get(accent, [0] * 16)

    # Extract audio features using librosa
    audio, sampling_rate = librosa.load(audio_path, sr=16000)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))

    features = gender + accent + [spectral_centroid, spectral_bandwidth, spectral_rolloff]
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=20)
    for el in mfcc:
        features.append(np.mean(el))

    features = np.array(features).reshape(1, -1)

    # Make predictions using the loaded model
    predicted_probs = model.predict(features)
    top_prediction_index = np.argmax(predicted_probs)
    top_prediction_class = age_classes[top_prediction_index]
    top_prediction_prob = predicted_probs[0][top_prediction_index]

    result = {
        'top_predicted_age_category': {
            'age_category': str(top_prediction_class) + " years old",
            'probability': float(top_prediction_prob)
        }
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
