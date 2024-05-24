from flask import Flask, render_template_string, request, jsonify
import librosa
import numpy as np
from keras.models import load_model
import pandas as pd
import os
import random

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
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&display=swap');
            body {
                font-family: "Roboto", sans-serif;
                background-color: #f0f0f0;
                margin: 0;
                padding: 0;
            }
            .container h1 {
                text-align: center;
                padding-top: 20px;
            }
            .container {
                max-width: 600px;
                margin: 50px auto;
                background-color: #f2f2f2;
                padding: 0 20px 20px 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(20, 43, 220, 0.2);
                display: none;
            }
            .nav {
                display: flex;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin: 0 -20px 20px -20px;
            }
            .nav div {
                flex: 1;
                font-size: large;
                font-weight: bold;
                text-align: center;
                border: 1px solid #e2e8f0;
                padding: 20px;
                cursor: pointer;
                background-color: #e2e8f0;
                transition: background-color 0.3s;
                box-shadow: 0 0 10px rgba(20, 43, 220, 0.2);
            }
            .nav div:hover {
                border-color: #000;
                background-color: #A5C4D4;
            }
            .nav .active {
                border-color: #000;
                background-color: #A5C4D4;
            }
            #about {
                display: block;
            }
            #about p{
                text-align: center;
                padding: 0 30px;
                font-size: 16px;
                font-weight: normal;
                color: #333;
                line-height: 1.6;
            }
            /* =========== Page One ===========*/
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                font-weight: bold;
                margin-bottom: 5px;
            }
            select, input[type="file"], button {
                width: 100%;
                padding: 8px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
                margin-top: 5px;
            }
            button {
                margin-top: 15px;
                background-color: #A5C4D4;
                border: 0 solid #e2e8f0;
                border-radius: 1.5rem;
                box-sizing: border-box;
                cursor: pointer;
                display: inline-block;
                padding: 1rem 1.6rem;
                color: #fff;
                text-align: center;
                text-transform: uppercase;
                transition: all .1s cubic-bezier(.4, 0, .2, 1);
                box-shadow: 0px 1px 2px rgba(166, 175, 195, 0.25);
                user-select: none;
                font-weight: bold;
                -webkit-user-select: none;
                touch-action: manipulation;
            }
            button:hover {
                background-color: #e2e8f0;
                color: #000;
            }
            .recording-btns{
                display: flex;
                gap: 10px;
            }
            .recording-btns button{
                padding: .5rem 1rem;
                width:10rem;
                border-radius: 5%;
                color: #e2e8f0;
                text-transform: capitalize;
                background-color: #7D9AA3;
            }
            .recording-btns button:hover{
                color: #fff;
                background-color: #506E76;
            }
            .sentence {
                margin: 20px 0;
                padding: 15px;
                background-color: #e0eaf1;
                border: 1px solid #ccc;
                border-radius: 8px;
            }
            .sentence h4 {
                margin: 0 0 10px 0;
                font-size: 15px;
                color: #333;
            }
            .sentence p {
                margin: 0;
                font-size: 16px;
                color: #555;
                text-transform: capitalize;
            }
            /* =========== Page Two ===========*/
            #results {display: none;}
            .container h1 {
                text-align: center;
                color: #333;
            }
            #ageResult, #ageResultProbability {
                font-size: 18px;
                text-align: center;
                color: #555;
            }
            #ageResult{
                margin-top: 60px;
            }
            #ageResultProbability {
                margin-top: 10px;
                margin-bottom: 60px;
                font-size: 16px;
                color: #777;
            }
        </style>
    </head>
    <body>
        <div class="container" id="about">
            <div class="nav" id="nav">
                <div class="gotoabout active" onclick="showAbout()">About</div>
                <div class="gotoprediction" onclick="showPrediction()">Age Prediction</div>
            </div>
            <h1>What is this App?</h1>
            <p>Utilizing advanced voice analysis techniques, this application determines the age range of individuals based on their voice samples. By incorporating information such as gender and accent, the predictions are tailored and more precise.</p>
        </div>
        <div class="container" id="questions">
            <div class="nav" id="nav">
                <div class="gotoabout" onclick="showAbout()">About</div>
                <div class="gotoprediction active" onclick="showPrediction()">Age Prediction</div>
            </div>
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
                        <option value="other">Other</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="audioFile">Upload an audio recording of your voice:</label>
                    <input type="file" id="audioFile" name="audioFile" accept="audio/mp3">
                </div>
                <button type="submit">Submit</button>
            </form>
        </div>
        <div class="container" id="results">
            <div class="nav" id="nav">
                <div class="gotoabout" onclick="showAbout()">About</div>
                <div class="gotoprediction active" onclick="showPrediction()">Age Prediction</div>
            </div>
            <h1>The results</h1>
            <div id="ageResult"></div>
            <div id="ageResultProbability"></div>
            <button id="tryAgainButton">Try Again?</button>
        </div>
        <script>
            function showAbout() {
                document.getElementById('about').style.display = 'block';
                document.getElementById('questions').style.display = 'none';
                document.getElementById('results').style.display = 'none';
                document.querySelector('.gotoabout').classList.add('active');
                document.querySelector('.gotoprediction').classList.remove('active');
            }
            function showPrediction() {
                document.getElementById('about').style.display = 'none';
                document.getElementById('questions').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.querySelector('.gotoabout').classList.remove('active');
                document.querySelector('.gotoprediction').classList.add('active');
            }
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
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content, sentence=sentence)

csv_file_path = 'newdata/features.csv'


def save_features_to_csv(features):
    data = {
        "features": [str(features)],
    }
    df = pd.DataFrame(data)

    if not os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, index=False, mode='w')
    else:
        df.to_csv(csv_file_path, index=False, mode='a', header=False)

@app.route('/predict', methods=['POST'])
def predict_age():
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
        'southatlandtic': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'other': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    if 'audioFile' in request.files:
        file = request.files['audioFile']
        audio_path = os.path.join('newdata', 'temp_audio.mp3')
        file.save(audio_path)
    else:
        return jsonify({'error': 'No audio file uploaded or recorded'})

    gender = request.form.get('gender')
    accent = request.form.get('accent')
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

    # Save features to the CSV file
    save_features_to_csv(features.tolist())


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
