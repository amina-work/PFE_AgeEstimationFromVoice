from flask import Flask, request, jsonify
import librosa
import numpy as np
from keras.models import load_model
import os

app = Flask(__name__)

age_classes = [1, 2, 3, 4, 5, 6, 7, 8]
model = load_model('my_model.h5')

@app.route('/')
def index():
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
        <div class="container">
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
                <div class="form-group">
                    <label for="audioFile">Upload an audio recording of your voice:</label>
                    <input type="file" id="audioFile" name="audioFile" accept="audio/mp3" required>
                </div>
                <button type="submit">Submit</button>
            </form>
            <div id="ageResult"></div><div id="ageResultProbability"></div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.route('/predict', methods=['POST'])
def predict_age():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['audioFile']
    gender = request.form.get('gender')
    accent = request.form.get('accent')
    
    # Ensure 'newdata' directory exists
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
    elif accent == "african":
        accent = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
            'age_category': int(top_prediction_class),
            'probability': float(top_prediction_prob)
        }
    }
    
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
