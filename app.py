from flask import Flask, render_template_string, request, jsonify
import random
from sklearn.preprocessing import OneHotEncoder
import librosa
import numpy as np
from keras.models import load_model
import os
import soundfile as sf
import audioread


app = Flask(__name__)

sentences = [
    "most of them were staring quietly at the big table",
    "you point to a trail of ants leading into the house",
    "the time must have been somewhere around six o'clock",
    "as i watched the planet seemed to grow larger and smaller",
    "i have the diet of a kid who found twenty dollars"
]

@app.route('/')
def index():
    random_sentence = random.choice(sentences)
    html_content = """
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Voice-Based Age Estimation</title>
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;500;600&family=Noto+Serif:wght@300;400;500;600;700&family=Roboto:wght@300;400;500;700&family=Rubik:ital,wght@0,300..900;1,300..900&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="{{ url_for('static', filename='app.css') }}">
    </head>
    <body>
        <div id="pageone" class="container">
            <h1>Estimate your age from just your voice</h1>
            <div class="gend_acc">
                <div>
                    <label for="gender">Gender:</label>
                    <select id="gender">
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                        <option value="other">Other</option>
                    </select>
                </div>
                <div>
                    <label for="accent">Accent:</label>
                    <select id="accent">
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
            </div>
            <div class="mic_sent">
                <div class="sent">
                    <h5>Read the following sentence:</h5>
                    <span id="randomSentence">{{ random_sentence }}</span>
                </div>
                <button id="micToggle" class="mic-toggle">
                    <span class="material-icons" id="micIcon">mic</span>
                </button>
            </div>
            <br />
            <p id="isRecording">Click the microphone to start recording</p>
            <audio src="" id="audioElement" controls></audio>
            <button class="nextpage" id="submitDetails">Submit</button>
        </div>

        <div id="pagethree" class="container">
            <h1>We estimate you are...</h1>
            <h2 class="result"></h2>
            <button class="goback" id="goback">Try Again?</button>
        </div>

        <script src="{{ url_for('static', filename='app.js') }}"></script>
    </body>
    </html>
    """
    return render_template_string(html_content, random_sentence=random_sentence)

gender_categories = ['male', 'female', 'other']
accent_categories = ['us', 'england', 'indian', 'australia', 'canada', 'scotland', 'african', 'newzealand', 'ireland', 'philippines', 'wales', 'bermuda', 'malaysia', 'singapore', 'hongkong', 'southatlandtic']
encoder = OneHotEncoder(categories=[gender_categories, accent_categories], sparse=False)
model = load_model('my_model.h5')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        features = []
        sampling_rate = 16000
        audio_file = request.files['audio']
        path = 'newdata/recording.mp3'

        # Ensure the directory exists
        if not os.path.exists('newdata'):
            os.makedirs('newdata')

        audio_file.save(path)
        
        # Load audio using audioread directly
        try:
            with audioread.audio_open(path) as f:
                audio = np.concatenate([np.frombuffer(buf, dtype=np.float32) for buf in f])
                sr = f.samplerate
        except Exception as e:
            print(f"Error reading audio with audioread: {e}")
            return jsonify({'error': f"Error reading audio: {e}"}), 500
        
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        
        gender = request.form.get('gender')
        accent = request.form.get('accent')

        try:
            encoded_features = encoder.transform([[gender, accent]]).flatten()
        except Exception as e:
            print(f"Error encoding features: {e}")
            return jsonify({'error': f"Error encoding features: {e}"}), 500
        
        features.extend(encoded_features)

        # Extract audio features
        try:
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return jsonify({'error': f"Error extracting audio features: {e}"}), 500
        
        features.extend([spectral_centroid, spectral_bandwidth, spectral_rolloff])

        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate)
            for el in mfcc:
                features.append(np.mean(el))
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return jsonify({'error': f"Error extracting MFCC: {e}"}), 500

        features_array = np.array(features).reshape(1, -1)
        
        # Predict age
        try:
            predicted_age = model.predict(features_array)
        except Exception as e:
            print(f"Error predicting age: {e}")
            return jsonify({'error': f"Error predicting age: {e}"}), 500

        return jsonify({'predicted_age': predicted_age[0][0]})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
