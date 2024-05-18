import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

age_classes = [1, 2, 3, 4, 5, 6, 7, 8]

model = load_model('my_model.h5')

audio_path = r"C:\Users\DrdrA\OneDrive\Desktop\UNI\M2\PFE\app\newdata\3_FAfrican2.mp3"
gender = [0, 1, 0]
accent = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

audio, sampling_rate = librosa.load(audio_path, sr=16000)  
features = []
features.extend(gender)
features.extend(accent)
spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))
features.append(spectral_centroid)
features.append(spectral_bandwidth)
features.append(spectral_rolloff)
print(features)
mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=20)  # Extract 20 MFCCs
for el in mfcc:
    features.append(np.mean(el))
print(features)
#scaler = StandardScaler()
features = np.array(features).reshape(1, -1)
print(features)
#features = scaler.fit_transform(features)

predicted_probs = model.predict(features)
predicted_probs = model.predict(features)
print("Predicted probabilities:", predicted_probs)

# Get the indices of the top two predicted classes
top_two_indices = np.argsort(predicted_probs)[0][-2:]

# Retrieve the top two age categories and their probabilities
top_two_classes = [age_classes[idx] for idx in top_two_indices]
top_two_probs = predicted_probs[0][top_two_indices]

# Print the results
print("Top two predicted age categories are:")
for i in range(len(top_two_classes)):
    print(f"{top_two_classes[i]} with probability {top_two_probs[i]}")

# Select the most likely age category
predicted_class_index = np.argmax(predicted_probs)
predicted_age = age_classes[predicted_class_index]
print("Predicted age category is:", predicted_age)