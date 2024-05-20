from flask import Flask, render_template_string, request, jsonify
import librosa
import numpy as np
from keras.models import load_model
import os
import random
from pydub import AudioSegment

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
                    <input type="file" id="audioFile" name="audioFile" accept="audio/mp3" required>
                </div>
                <div class="form-group">
                    <label for="recordAudio">Or record an audio recording of your voice:</label>
                    <div class="recording-btns">
                        <audio id="audioPlayback" controls></audio>
                        <button type="button" id="recordButton">Record</button>
                        <button type="button" id="stopButton" disabled>Stop</button>
                    </div>
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
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content, sentence=sentence)

@app.route('/predict', methods=['POST'])
def predict_age():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['audioFile']
    gender = request.form.get('gender')
    accent = request.form.get('accent')
    
    if not os.path.exists('newdata'):
        os.makedirs('newdata')
    
    audio_path = os.path.join('newdata', 'temp_audio.mp3')
    file.save(audio_path)

    if gender == "female":
        gender = [0, 1, 0]
    elif gender == "male":
        gender = [1, 0, 0]
    else: 
        gender = [0, 0, 0]

    if accent == "us":
        accent = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif accent == "england":
        accent = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif accent == "indian":
        accent = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif accent == "australia":
        accent = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif accent == "canada":
        accent = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif accent == "scotland":
        accent = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif accent == "african":
        accent = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif accent == "newzealand":
        accent = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif accent == "ireland":
        accent = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif accent == "philippines":
        accent = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif accent == "wales":
        accent = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif accent == "bermuda":
        accent = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif accent == "malaysia":
        accent = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif accent == "singapore":
        accent = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif accent == "hongkong":
        accent = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif accent == "southatlandtic":
        accent = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    features = gender + accent
    
    audio, sampling_rate = librosa.load(audio_path, sr=16000)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))
    features.extend([spectral_centroid, spectral_bandwidth, spectral_rolloff])
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=20)
    for el in mfcc:
        features.append(np.mean(el))
    
    features = np.array(features).reshape(1, -1)
    
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
    
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)